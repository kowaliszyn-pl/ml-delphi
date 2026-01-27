program Gpt2Inference;

{$APPTYPE CONSOLE}

uses
  SysUtils,
  Windows,
  Math,
  System.Generics.Collections,
  MatrixUtility in 'MatrixUtility.pas',
  Gpt2HParams in 'Gpt2HParams.pas',
  Gpt2Params in 'Gpt2Params.pas',
  Gpt2Encoder in 'Gpt2Encoder.pas';

const
  NegativeInfinity: Single = -1e10;
  ModelSize = '124M';
  ModelsDir = '..\..\..\..\..\..\data\GPT-2\';
  NumTokensToGenerate = 100;
  Seed = 42;
  WithProbabilities = True;

type
  TGenerateOptions = record
    Temperature: Single;
    TopK: Integer;
    TopP: Single;
  end;

  TCandidate = record
    TokenId: Integer;
    Probability: Single;
  end;

  TCandidateArray = array of TCandidate;

  TGenerateResult = record
    TokenId: Integer;
    Candidates: TCandidateArray;
  end;

var
  Deterministic: TGenerateOptions;
  LittleFreedom: TGenerateOptions;
  Nondeterministic: TGenerateOptions;
  Crazy: TGenerateOptions;
  Wise: TGenerateOptions;

procedure InitializeOptions;
begin
  Deterministic.Temperature := 1.0;
  Deterministic.TopK := 1;
  Deterministic.TopP := 1.0;

  LittleFreedom.Temperature := 1.0;
  LittleFreedom.TopK := 5;
  LittleFreedom.TopP := 1.0;

  Nondeterministic.Temperature := 1.0;
  Nondeterministic.TopK := 40;
  Nondeterministic.TopP := 1.0;

  Crazy.Temperature := 1.3;
  Crazy.TopK := 40;
  Crazy.TopP := 0.9;

  Wise.Temperature := 0.6;
  Wise.TopK := 30;
  Wise.TopP := 0.75;
end;

{ Additional Matrix Functions }

function GetRow(const A: TMatrix2D; rowIndex: Integer): TMatrix1D;
var
  cols, j: Integer;
begin
  cols := Length(A[0]);
  SetLength(Result, cols);
  for j := 0 to cols - 1 do
    Result[j] := A[rowIndex, j];
end;

function SoftmaxStable(const A: TMatrix2D): TMatrix2D;
var
  rows, cols, i, j: Integer;
  maxVal, sumExp: Single;
begin
  rows := Length(A);
  cols := Length(A[0]);
  Result := CreateMatrix2D(rows, cols);

  for i := 0 to rows - 1 do
  begin
    { Find max for numerical stability }
    maxVal := A[i, 0];
    for j := 1 to cols - 1 do
      if A[i, j] > maxVal then
        maxVal := A[i, j];

    { Compute exp and sum }
    sumExp := 0;
    for j := 0 to cols - 1 do
    begin
      Result[i, j] := Exp(A[i, j] - maxVal);
      sumExp := sumExp + Result[i, j];
    end;

    { Normalize }
    for j := 0 to cols - 1 do
      Result[i, j] := Result[i, j] / sumExp;
  end;
end;

function SoftmaxStableWithTemperature(const A: TMatrix1D; temperature: Single): TMatrix1D;
var
  len, i: Integer;
  maxVal, sumExp: Single;
begin
  len := Length(A);
  SetLength(Result, len);

  { Find max for numerical stability }
  maxVal := A[0];
  for i := 1 to len - 1 do
    if A[i] > maxVal then
      maxVal := A[i];

  { Compute exp with temperature and sum }
  sumExp := 0;
  for i := 0 to len - 1 do
  begin
    Result[i] := Exp((A[i] - maxVal) / temperature);
    sumExp := sumExp + Result[i];
  end;

  { Normalize }
  for i := 0 to len - 1 do
    Result[i] := Result[i] / sumExp;
end;

function Gelu(const A: TMatrix2D): TMatrix2D;
var
  rows, cols, i, j: Integer;
  x: Single;
begin
  rows := Length(A);
  cols := Length(A[0]);
  Result := CreateMatrix2D(rows, cols);

  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do
    begin
      x := A[i, j];
      { GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) }
      Result[i, j] := 0.5 * x * (1 + Tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    end;
end;

function StandardizeByRows(const A: TMatrix2D): TMatrix2D;
var
  rows, cols, i, j: Integer;
  mean, variance, stdDev, value: Single;
const
  Epsilon: Single = 1e-5;
begin
  rows := Length(A);
  cols := Length(A[0]);
  Result := CreateMatrix2D(rows, cols);

  for i := 0 to rows - 1 do
  begin
    { Compute mean }
    mean := 0;
    for j := 0 to cols - 1 do
      mean := mean + A[i, j];
    mean := mean / cols;

    { Compute variance }
    variance := 0;
    for j := 0 to cols - 1 do
    begin
      value := A[i, j] - mean;
      variance := variance + value * value;
    end;
    variance := variance / cols;
    stdDev := Sqrt(variance + Epsilon);

    { Normalize }
    for j := 0 to cols - 1 do
      Result[i, j] := (A[i, j] - mean) / stdDev;
  end;
end;

function AddMatrix(const A, B: TMatrix2D): TMatrix2D;
var
  rows, cols, i, j: Integer;
begin
  rows := Length(A);
  cols := Length(A[0]);
  Result := CreateMatrix2D(rows, cols);
  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do
      Result[i, j] := A[i, j] + B[i, j];
end;

{ GPT-2 Forward Pass Functions }

function EmbedTokens(const inputTokenIds: TIntArray;
  const tokenEmbeddings, positionalEmbeddings: TMatrix2D): TMatrix2D;
var
  inputTokens, embeddingSize, positionInInputSequence, tokenId, embeddingIndex: Integer;
  value: Single;
begin
  inputTokens := Length(inputTokenIds);
  embeddingSize := Length(tokenEmbeddings[0]);
  Result := CreateMatrix2D(inputTokens, embeddingSize);

  for positionInInputSequence := 0 to inputTokens - 1 do
  begin
    tokenId := inputTokenIds[positionInInputSequence];
    if (tokenId < 0) or (tokenId >= Length(tokenEmbeddings)) then
      raise Exception.CreateFmt('Token id %d is outside the vocabulary range.', [tokenId]);

    for embeddingIndex := 0 to embeddingSize - 1 do
    begin
      value := tokenEmbeddings[tokenId, embeddingIndex];
      value := value + positionalEmbeddings[positionInInputSequence, embeddingIndex];
      Result[positionInInputSequence, embeddingIndex] := value;
    end;
  end;
end;

function LinearForward(const x: TMatrix2D; const linearParams: TGpt2LinearParams): TMatrix2D;
begin
  Result := AddRow(MultiplyDot(x, linearParams.Weights), linearParams.Bias);
end;

function LayerNormForward(const x: TMatrix2D; const layerNorm: TGpt2LayerNormParams): TMatrix2D;
var
  normalized: TMatrix2D;
begin
  normalized := StandardizeByRows(x);
  Result := AddRow(MultiplyElementwise(normalized, layerNorm.Gamma), layerNorm.Beta);
end;

procedure SplitIntoQKV(const x: TMatrix2D; out q, k, v: TMatrix2D);
var
  nSeq, nEmbd, i, j: Integer;
begin
  nSeq := Length(x);
  nEmbd := Length(x[0]) div 3;
  q := CreateMatrix2D(nSeq, nEmbd);
  k := CreateMatrix2D(nSeq, nEmbd);
  v := CreateMatrix2D(nSeq, nEmbd);

  for i := 0 to nSeq - 1 do
    for j := 0 to nEmbd - 1 do
    begin
      q[i, j] := x[i, j];
      k[i, j] := x[i, j + nEmbd];
      v[i, j] := x[i, j + 2 * nEmbd];
    end;
end;

type
  TMatrix3D = array of array of array of Single;

function SplitHeads(const qkv: TMatrix2D; headCount: Integer): TMatrix3D;
var
  nSeq, nEmbd, headDim, i, h, j: Integer;
begin
  nSeq := Length(qkv);
  nEmbd := Length(qkv[0]);
  if nEmbd mod headCount <> 0 then
    raise Exception.Create('Embedding size must be divisible by head count.');
  headDim := nEmbd div headCount;

  SetLength(Result, headCount, nSeq, headDim);
  for i := 0 to nSeq - 1 do
    for h := 0 to headCount - 1 do
      for j := 0 to headDim - 1 do
        Result[h, i, j] := qkv[i, h * headDim + j];
end;

function GetHead(const heads: TMatrix3D; headIndex: Integer): TMatrix2D;
var
  nSeq, headDim, i, j: Integer;
begin
  nSeq := Length(heads[0]);
  headDim := Length(heads[0, 0]);
  Result := CreateMatrix2D(nSeq, headDim);
  for i := 0 to nSeq - 1 do
    for j := 0 to headDim - 1 do
      Result[i, j] := heads[headIndex, i, j];
end;

function BuildCausalMask(sequenceLength: Integer): TMatrix2D;
var
  row, col: Integer;
begin
  Result := CreateMatrix2D(sequenceLength, sequenceLength);
  for row := 0 to sequenceLength - 1 do
    for col := 0 to sequenceLength - 1 do
      if col <= row then
        Result[row, col] := 0
      else
        Result[row, col] := NegativeInfinity;
end;

function Attention(const queryHeadMatrix, keyHeadMatrix, valueHeadMatrix, causalMask: TMatrix2D): TMatrix2D;
var
  keyEmbeddingWidth: Integer;
  inverseSqrtKeyDim: Single;
  rawAttentionScores, attentionProbabilities: TMatrix2D;
begin
  keyEmbeddingWidth := Length(queryHeadMatrix[0]);
  inverseSqrtKeyDim := 1.0 / Sqrt(keyEmbeddingWidth);

  rawAttentionScores := MultiplyDot(queryHeadMatrix, Transpose(keyHeadMatrix));
  rawAttentionScores := Multiply(rawAttentionScores, inverseSqrtKeyDim);
  rawAttentionScores := AddMatrix(rawAttentionScores, causalMask);
  attentionProbabilities := SoftmaxStable(rawAttentionScores);
  Result := MultiplyDot(attentionProbabilities, valueHeadMatrix);
end;

function MultiHeadAttention(const x: TMatrix2D; const attention: TGpt2MultiHeadAttentionParams;
  headCount: Integer): TMatrix2D;
var
  projected: TMatrix2D;
  q, k, v: TMatrix2D;
  qHeads, kHeads, vHeads: TMatrix3D;
  inputSequenceLength, headDim, headIndex, i, j, h: Integer;
  causalMask: TMatrix2D;
  outHeads: TMatrix3D;
  qh, kh, vh, attn, mergedHeads: TMatrix2D;
begin
  projected := LinearForward(x, attention.Projection);
  SplitIntoQKV(projected, q, k, v);

  qHeads := SplitHeads(q, headCount);
  kHeads := SplitHeads(k, headCount);
  vHeads := SplitHeads(v, headCount);

  inputSequenceLength := Length(projected);
  causalMask := BuildCausalMask(inputSequenceLength);

  headDim := Length(qHeads[0, 0]);
  SetLength(outHeads, headCount, inputSequenceLength, headDim);

  for headIndex := 0 to headCount - 1 do
  begin
    qh := GetHead(qHeads, headIndex);
    kh := GetHead(kHeads, headIndex);
    vh := GetHead(vHeads, headIndex);
    attn := Attention(qh, kh, vh, causalMask);

    for i := 0 to inputSequenceLength - 1 do
      for j := 0 to headDim - 1 do
        outHeads[headIndex, i, j] := attn[i, j];
  end;

  mergedHeads := CreateMatrix2D(inputSequenceLength, headCount * headDim);
  for i := 0 to inputSequenceLength - 1 do
    for h := 0 to headCount - 1 do
      for j := 0 to headDim - 1 do
        mergedHeads[i, h * headDim + j] := outHeads[h, i, j];

  Result := LinearForward(mergedHeads, attention.OutputProjection);
end;

function FeedForwardNetwork(const x: TMatrix2D; const mlp: TGpt2MultiLayerPerceptron): TMatrix2D;
var
  hiddenLayer: TMatrix2D;
begin
  hiddenLayer := LinearForward(x, mlp.FullyConnected);
  hiddenLayer := Gelu(hiddenLayer);
  Result := LinearForward(hiddenLayer, mlp.OutputProjection);
end;

function TransformerBlockForward(const x: TMatrix2D; const block: TGpt2Block;
  headCount: Integer): TMatrix2D;
var
  normalizedXForAttention, attentionOutput: TMatrix2D;
  normalizedXForFeedForward, feedForwardOutput: TMatrix2D;
begin
  normalizedXForAttention := LayerNormForward(x, block.LayerNorm1);
  attentionOutput := MultiHeadAttention(normalizedXForAttention, block.Attention, headCount);
  Result := AddMatrix(x, attentionOutput);

  normalizedXForFeedForward := LayerNormForward(Result, block.LayerNorm2);
  feedForwardOutput := FeedForwardNetwork(normalizedXForFeedForward, block.MultiLayerPerceptron);
  Result := AddMatrix(Result, feedForwardOutput);
end;

function Forward(const inputIds: TIntArray; const modelParams: TGpt2Params;
  headCount: Integer): TMatrix2D;
var
  x: TMatrix2D;
  blockIndex: Integer;
begin
  x := EmbedTokens(inputIds, modelParams.TokenEmbeddings, modelParams.PositionalEmbeddings);

  for blockIndex := 0 to High(modelParams.Blocks) do
    x := TransformerBlockForward(x, modelParams.Blocks[blockIndex], headCount);

  x := LayerNormForward(x, modelParams.FinalLayerNorm);
  Result := MultiplyDot(x, Transpose(modelParams.TokenEmbeddings));
end;

function Generate(const inputIds: TIntArray; const modelParams: TGpt2Params;
  headCount, nTokensToGenerate: Integer; const options: TGenerateOptions): TList<TGenerateResult>;
var
  inputs: TList<Integer>;
  i, j, nextId, cutOffIndex, tokenIndex: Integer;
  logits: TMatrix2D;
  lastTokenLogits, softmaxedLogits: TMatrix1D;
  tokenList: TList<TCandidate>;
  candidate: TCandidate;
  cumulativeProbability, sumProbability, sample, cumulative: Single;
  genResult: TGenerateResult;
  inputArray: TIntArray;
begin
  Result := TList<TGenerateResult>.Create;
  inputs := TList<Integer>.Create;
  tokenList := TList<TCandidate>.Create;
  try
    for i := 0 to High(inputIds) do
      inputs.Add(inputIds[i]);

    for i := 0 to nTokensToGenerate - 1 do
    begin
      SetLength(inputArray, inputs.Count);
      for j := 0 to inputs.Count - 1 do
        inputArray[j] := inputs[j];

      logits := Forward(inputArray, modelParams, headCount);
      lastTokenLogits := GetRow(logits, Length(logits) - 1);
      softmaxedLogits := SoftmaxStableWithTemperature(lastTokenLogits, options.Temperature);

      { Create token list with probabilities }
      tokenList.Clear;
      for j := 0 to High(softmaxedLogits) do
      begin
        candidate.TokenId := j;
        candidate.Probability := softmaxedLogits[j];
        tokenList.Add(candidate);
      end;

      { Sort descending by probability }
      tokenList.Sort(TComparer<TCandidate>.Construct(
        function(const Left, Right: TCandidate): Integer
        begin
          if Right.Probability > Left.Probability then
            Result := 1
          else if Right.Probability < Left.Probability then
            Result := -1
          else
            Result := 0;
        end));

      { Apply top-K filtering }
      if (options.TopK > 0) and (tokenList.Count > options.TopK) then
        while tokenList.Count > options.TopK do
          tokenList.Delete(tokenList.Count - 1);

      { Apply nucleus (top-P) filtering }
      if options.TopP < 1.0 then
      begin
        cumulativeProbability := 0;
        cutOffIndex := tokenList.Count;
        for j := 0 to tokenList.Count - 1 do
        begin
          cumulativeProbability := cumulativeProbability + tokenList[j].Probability;
          if cumulativeProbability >= options.TopP then
          begin
            cutOffIndex := j + 1;
            Break;
          end;
        end;
        while tokenList.Count > cutOffIndex do
          tokenList.Delete(tokenList.Count - 1);
      end;

      { Sample from distribution }
      sumProbability := 0;
      for j := 0 to tokenList.Count - 1 do
        sumProbability := sumProbability + tokenList[j].Probability;

      sample := Random * sumProbability;
      cumulative := 0;
      nextId := tokenList[0].TokenId;

      for tokenIndex := 0 to tokenList.Count - 1 do
      begin
        cumulative := cumulative + tokenList[tokenIndex].Probability;
        if cumulative >= sample then
        begin
          nextId := tokenList[tokenIndex].TokenId;
          Break;
        end;
      end;

      { Build result }
      genResult.TokenId := nextId;
      SetLength(genResult.Candidates, Min(5, tokenList.Count));
      for j := 0 to High(genResult.Candidates) do
      begin
        genResult.Candidates[j].TokenId := tokenList[j].TokenId;
        genResult.Candidates[j].Probability := tokenList[j].Probability;
      end;

      inputs.Add(nextId);
      Result.Add(genResult);
    end;
  finally
    inputs.Free;
    tokenList.Free;
  end;
end;

var
  hParams: TGpt2HParams;
  encoder: TGpt2Encoder;
  modelParams: TGpt2Params;
  modelDirectory, prompt, nextWord, candidatesStr: string;
  createDummy: Boolean;
  inputIds, outputTokenIds: TIntArray;
  results: TList<TGenerateResult>;
  genResult: TGenerateResult;
  i: Integer;

begin
  SetConsoleOutputCP(CP_UTF8);
  Randomize;
  RandSeed := Seed;
  InitializeOptions;

  Writeln(Format('NumTokensToGenerate: %d', [NumTokensToGenerate]));
  Writeln(Format('ModelSize: %s', [ModelSize]));
  Writeln(Format('ModelsDir: %s', [ModelsDir]));

  { Prepare the model }
  createDummy := ModelSize = '0';

  if createDummy then
  begin
    hParams := CreateDefaultHParams;
    encoder := TGpt2Encoder.CreateDummy(hParams);
    modelParams := CreateRandomGpt2Params(hParams, Seed);
  end
  else
  begin
    modelDirectory := IncludeTrailingPathDelimiter(ModelsDir) + ModelSize;
    hParams := LoadHParamsFromDirectory(modelDirectory);
    encoder := TGpt2Encoder.FromDirectory(modelDirectory);
    modelParams := LoadGpt2ParamsFromDirectory(modelDirectory, hParams);
  end;

  try
    while True do
    begin
      Write('Enter prompt: ');
      SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED or FOREGROUND_GREEN);
      Readln(prompt);
      SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED or FOREGROUND_GREEN or FOREGROUND_BLUE);

      if prompt = '' then
        Break;

      inputIds := encoder.Encode(prompt);

      if Length(inputIds) + NumTokensToGenerate >= hParams.ContextSize then
        raise Exception.Create('Input prompt is too long for the model''s context size.');

      results := Generate(inputIds, modelParams, hParams.HeadCount, NumTokensToGenerate, Wise);
      try
        for genResult in results do
        begin
          SetLength(outputTokenIds, 1);
          outputTokenIds[0] := genResult.TokenId;
          nextWord := encoder.Decode(outputTokenIds);

          if WithProbabilities then
          begin
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED or FOREGROUND_GREEN);
            Write(Format('%s ', [nextWord]));
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED or FOREGROUND_GREEN or FOREGROUND_BLUE);

            candidatesStr := '';
            for i := 0 to High(genResult.Candidates) do
            begin
              SetLength(outputTokenIds, 1);
              outputTokenIds[0] := genResult.Candidates[i].TokenId;
              if i > 0 then
                candidatesStr := candidatesStr + ', ';
              candidatesStr := candidatesStr + Format('''%s'' - %.2f%%',
                [encoder.Decode(outputTokenIds), genResult.Candidates[i].Probability * 100]);
            end;
            Writeln(Format('[%s]', [candidatesStr]));
          end
          else
          begin
            Write(nextWord);
          end;
        end;
      finally
        results.Free;
      end;

      Writeln;
      Writeln;
    end;
  finally
    encoder.Free;
  end;

  Writeln;
  Writeln('Press ENTER...');
  Readln;
end.
