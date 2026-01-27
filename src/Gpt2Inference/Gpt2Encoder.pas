unit Gpt2Encoder;

interface

uses
  Gpt2HParams;

type
  TIntArray = array of Integer;

  TGpt2Encoder = class
  private
    FEncoder: array of record
      Token: string;
      Id: Integer;
    end;
    FDecoder: array of record
      Id: Integer;
      Token: string;
    end;
    FByteEncoder: array[0..255] of string;
    FByteDecoder: array of record
      Ch: Char;
      Value: Byte;
    end;
    FBpeRanks: array of record
      First: string;
      Second: string;
      Rank: Integer;
    end;
    FCache: array of record
      Key: string;
      Value: string;
    end;

    function FindEncoderToken(const token: string): Integer;
    function FindDecoderToken(id: Integer): string;
    function FindByteDecoderValue(ch: Char): Byte;
    function GetBpeRank(const first, second: string): Integer;
    function GetCachedBpe(const token: string; out value: string): Boolean;
    procedure AddToCache(const key, value: string);
    procedure BuildByteUnicodeLookups;
    function EncodeUtf8(const token: string): string;
    function ApplyBpe(const token: string): string;
    function MatchTokenPattern(const text: string): TArray<string>;
  public
    constructor Create;
    destructor Destroy; override;

    class function CreateDummy(const hParams: TGpt2HParams): TGpt2Encoder;
    class function FromDirectory(const modelDirectory: string): TGpt2Encoder;

    function Encode(const text: string): TIntArray;
    function Decode(const tokens: TIntArray): string;
  end;

implementation

uses
  SysUtils, Classes, System.JSON, System.RegularExpressions,
  System.Generics.Collections;

constructor TGpt2Encoder.Create;
begin
  inherited Create;
  BuildByteUnicodeLookups;
end;

destructor TGpt2Encoder.Destroy;
begin
  inherited Destroy;
end;

procedure TGpt2Encoder.BuildByteUnicodeLookups;
var
  bs: TList<Integer>;
  cs: TList<Integer>;
  existing: TDictionary<Integer, Boolean>;
  b, n, i: Integer;
begin
  bs := TList<Integer>.Create;
  cs := TList<Integer>.Create;
  existing := TDictionary<Integer, Boolean>.Create;
  try
    { Add printable ASCII and extended Latin characters }
    for b := Ord('!') to Ord('~') do
    begin
      bs.Add(b);
      cs.Add(b);
      existing.Add(b, True);
    end;
    for b := Ord('¡') to Ord('¬') do
    begin
      bs.Add(b);
      cs.Add(b);
      existing.Add(b, True);
    end;
    for b := Ord('®') to Ord('ÿ') do
    begin
      bs.Add(b);
      cs.Add(b);
      existing.Add(b, True);
    end;

    { Add remaining bytes mapped to higher Unicode }
    n := 0;
    for b := 0 to 255 do
    begin
      if not existing.ContainsKey(b) then
      begin
        bs.Add(b);
        cs.Add(256 + n);
        Inc(n);
      end;
    end;

    { Build lookup tables }
    SetLength(FByteDecoder, bs.Count);
    for i := 0 to bs.Count - 1 do
    begin
      FByteEncoder[bs[i]] := Char(cs[i]);
      FByteDecoder[i].Ch := Char(cs[i]);
      FByteDecoder[i].Value := Byte(bs[i]);
    end;
  finally
    bs.Free;
    cs.Free;
    existing.Free;
  end;
end;

function TGpt2Encoder.FindEncoderToken(const token: string): Integer;
var
  i: Integer;
begin
  for i := 0 to High(FEncoder) do
    if FEncoder[i].Token = token then
      Exit(FEncoder[i].Id);
  Result := -1;
end;

function TGpt2Encoder.FindDecoderToken(id: Integer): string;
var
  i: Integer;
begin
  for i := 0 to High(FDecoder) do
    if FDecoder[i].Id = id then
      Exit(FDecoder[i].Token);
  Result := '';
end;

function TGpt2Encoder.FindByteDecoderValue(ch: Char): Byte;
var
  i: Integer;
begin
  for i := 0 to High(FByteDecoder) do
    if FByteDecoder[i].Ch = ch then
      Exit(FByteDecoder[i].Value);
  raise Exception.CreateFmt('Character ''%s'' missing from byte decoder.', [ch]);
end;

function TGpt2Encoder.GetBpeRank(const first, second: string): Integer;
var
  i: Integer;
begin
  for i := 0 to High(FBpeRanks) do
    if (FBpeRanks[i].First = first) and (FBpeRanks[i].Second = second) then
      Exit(FBpeRanks[i].Rank);
  Result := MaxInt;
end;

function TGpt2Encoder.GetCachedBpe(const token: string; out value: string): Boolean;
var
  i: Integer;
begin
  for i := 0 to High(FCache) do
    if FCache[i].Key = token then
    begin
      value := FCache[i].Value;
      Exit(True);
    end;
  Result := False;
end;

procedure TGpt2Encoder.AddToCache(const key, value: string);
var
  len: Integer;
begin
  len := Length(FCache);
  SetLength(FCache, len + 1);
  FCache[len].Key := key;
  FCache[len].Value := value;
end;

class function TGpt2Encoder.CreateDummy(const hParams: TGpt2HParams): TGpt2Encoder;
var
  i: Integer;
begin
  Result := TGpt2Encoder.Create;
  SetLength(Result.FEncoder, hParams.VocabularySize);
  SetLength(Result.FDecoder, hParams.VocabularySize);
  for i := 0 to hParams.VocabularySize - 1 do
  begin
    Result.FEncoder[i].Token := Format('token_%d', [i]);
    Result.FEncoder[i].Id := i;
    Result.FDecoder[i].Id := i;
    Result.FDecoder[i].Token := Format('token_%d', [i]);
  end;
end;

class function TGpt2Encoder.FromDirectory(const modelDirectory: string): TGpt2Encoder;
var
  encoderPath, mergesPath: string;
  encoderJson: string;
  jsonObj: TJSONObject;
  pair: TJSONPair;
  mergeLines: TStringList;
  i, idx: Integer;
  parts: TArray<string>;
begin
  Result := TGpt2Encoder.Create;

  encoderPath := IncludeTrailingPathDelimiter(modelDirectory) + 'encoder.json';
  mergesPath := IncludeTrailingPathDelimiter(modelDirectory) + 'vocab.bpe';

  { Load encoder }
  mergeLines := TStringList.Create;
  try
    mergeLines.LoadFromFile(encoderPath);
    encoderJson := mergeLines.Text;
  finally
    mergeLines.Free;
  end;

  jsonObj := TJSONObject.ParseJSONValue(encoderJson) as TJSONObject;
  try
    if jsonObj = nil then
      raise Exception.Create('Failed to parse encoder.json');

    SetLength(Result.FEncoder, jsonObj.Count);
    SetLength(Result.FDecoder, jsonObj.Count);
    idx := 0;
    for pair in jsonObj do
    begin
      Result.FEncoder[idx].Token := pair.JsonString.Value;
      Result.FEncoder[idx].Id := (pair.JsonValue as TJSONNumber).AsInt;
      Result.FDecoder[idx].Id := (pair.JsonValue as TJSONNumber).AsInt;
      Result.FDecoder[idx].Token := pair.JsonString.Value;
      Inc(idx);
    end;
  finally
    jsonObj.Free;
  end;

  { Load merges }
  mergeLines := TStringList.Create;
  try
    mergeLines.LoadFromFile(mergesPath);
    idx := 0;
    for i := 1 to mergeLines.Count - 1 do
    begin
      if Trim(mergeLines[i]) = '' then
        Continue;
      parts := mergeLines[i].Split([' '], TStringSplitOptions.ExcludeEmpty);
      if Length(parts) = 2 then
      begin
        SetLength(Result.FBpeRanks, idx + 1);
        Result.FBpeRanks[idx].First := parts[0];
        Result.FBpeRanks[idx].Second := parts[1];
        Result.FBpeRanks[idx].Rank := idx;
        Inc(idx);
      end;
    end;
  finally
    mergeLines.Free;
  end;
end;

function TGpt2Encoder.EncodeUtf8(const token: string): string;
var
  utf8Bytes: TBytes;
  i: Integer;
begin
  utf8Bytes := TEncoding.UTF8.GetBytes(token);
  Result := '';
  for i := 0 to High(utf8Bytes) do
    Result := Result + FByteEncoder[utf8Bytes[i]];
end;

function TGpt2Encoder.MatchTokenPattern(const text: string): TArray<string>;
var
  regex: TRegEx;
  matches: TMatchCollection;
  i: Integer;
begin
  regex := TRegEx.Create('''s|''t|''re|''ve|''m|''ll|''d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+');
  matches := regex.Matches(text);
  SetLength(Result, matches.Count);
  for i := 0 to matches.Count - 1 do
    Result[i] := matches[i].Value;
end;

function TGpt2Encoder.ApplyBpe(const token: string): string;
var
  word: TList<string>;
  i, j, bestIdx, bestRank, rank: Integer;
  cached: string;
  pairs: TList<TPair<string, string>>;
  first, second, merged: string;
  newWord: TList<string>;
  found: Boolean;
begin
  if GetCachedBpe(token, cached) then
    Exit(cached);

  word := TList<string>.Create;
  pairs := TList<TPair<string, string>>.Create;
  newWord := TList<string>.Create;
  try
    { Initialize word as individual characters }
    for i := 1 to Length(token) do
      word.Add(token[i]);

    if word.Count < 2 then
    begin
      Result := token;
      AddToCache(token, Result);
      Exit;
    end;

    while True do
    begin
      { Get all pairs }
      pairs.Clear;
      for i := 0 to word.Count - 2 do
        pairs.Add(TPair<string, string>.Create(word[i], word[i + 1]));

      if pairs.Count = 0 then
        Break;

      { Find best pair by rank }
      bestIdx := -1;
      bestRank := MaxInt;
      for i := 0 to pairs.Count - 1 do
      begin
        rank := GetBpeRank(pairs[i].Key, pairs[i].Value);
        if rank < bestRank then
        begin
          bestRank := rank;
          bestIdx := i;
        end;
      end;

      if bestRank = MaxInt then
        Break;

      first := pairs[bestIdx].Key;
      second := pairs[bestIdx].Value;
      merged := first + second;

      { Apply merge }
      newWord.Clear;
      i := 0;
      while i < word.Count do
      begin
        if (i < word.Count - 1) and (word[i] = first) and (word[i + 1] = second) then
        begin
          newWord.Add(merged);
          Inc(i, 2);
        end
        else
        begin
          newWord.Add(word[i]);
          Inc(i);
        end;
      end;

      word.Clear;
      for i := 0 to newWord.Count - 1 do
        word.Add(newWord[i]);

      if word.Count = 1 then
        Break;
    end;

    Result := '';
    for i := 0 to word.Count - 1 do
    begin
      if i > 0 then
        Result := Result + ' ';
      Result := Result + word[i];
    end;

    AddToCache(token, Result);
  finally
    word.Free;
    pairs.Free;
    newWord.Free;
  end;
end;

function TGpt2Encoder.Encode(const text: string): TIntArray;
var
  matches: TArray<string>;
  i, j, tokenId: Integer;
  encoded, bpeResult: string;
  bpeTokens: TArray<string>;
  resultList: TList<Integer>;
begin
  resultList := TList<Integer>.Create;
  try
    matches := MatchTokenPattern(text);
    for i := 0 to High(matches) do
    begin
      encoded := EncodeUtf8(matches[i]);
      bpeResult := ApplyBpe(encoded);
      bpeTokens := bpeResult.Split([' '], TStringSplitOptions.ExcludeEmpty);
      for j := 0 to High(bpeTokens) do
      begin
        tokenId := FindEncoderToken(bpeTokens[j]);
        if tokenId < 0 then
          raise Exception.CreateFmt('Token ''%s'' not present in vocabulary.', [bpeTokens[j]]);
        resultList.Add(tokenId);
      end;
    end;

    SetLength(Result, resultList.Count);
    for i := 0 to resultList.Count - 1 do
      Result[i] := resultList[i];
  finally
    resultList.Free;
  end;
end;

function TGpt2Encoder.Decode(const tokens: TIntArray): string;
var
  i: Integer;
  textBuilder: string;
  piece: string;
  byteBuffer: TList<Byte>;
  bytes: TBytes;
begin
  textBuilder := '';
  for i := 0 to High(tokens) do
  begin
    piece := FindDecoderToken(tokens[i]);
    if piece = '' then
      raise Exception.CreateFmt('Token id ''%d'' not present in decoder.', [tokens[i]]);
    textBuilder := textBuilder + piece;
  end;

  byteBuffer := TList<Byte>.Create;
  try
    for i := 1 to Length(textBuilder) do
      byteBuffer.Add(FindByteDecoderValue(textBuilder[i]));

    SetLength(bytes, byteBuffer.Count);
    for i := 0 to byteBuffer.Count - 1 do
      bytes[i] := byteBuffer[i];

    Result := TEncoding.UTF8.GetString(bytes);
  finally
    byteBuffer.Free;
  end;
end;

end.
