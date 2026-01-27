unit Gpt2HParams;

interface

type
  TGpt2HParams = record
    ContextSize: Integer;      // n_ctx
    HeadCount: Integer;        // n_head
    VocabularySize: Integer;   // n_vocab
    EmbeddingSize: Integer;    // n_embd
    LayerCount: Integer;       // n_layer
    function HeadSize: Integer;
  end;

function CreateDefaultHParams: TGpt2HParams;
function LoadHParamsFromDirectory(const modelDirectory: string): TGpt2HParams;

implementation

uses
  SysUtils, Classes, System.JSON;

function TGpt2HParams.HeadSize: Integer;
begin
  if HeadCount <> 0 then
    Result := EmbeddingSize div HeadCount
  else
    Result := 0;
end;

function CreateDefaultHParams: TGpt2HParams;
begin
  Result.ContextSize := 1024;
  Result.HeadCount := 12;
  Result.VocabularySize := 50257;
  Result.EmbeddingSize := 768;
  Result.LayerCount := 12;
end;

function LoadHParamsFromDirectory(const modelDirectory: string): TGpt2HParams;
var
  path: string;
  jsonText: string;
  jsonObj: TJSONObject;
  lines: TStringList;
begin
  path := IncludeTrailingPathDelimiter(modelDirectory) + 'hparams.json';
  lines := TStringList.Create;
  try
    lines.LoadFromFile(path);
    jsonText := lines.Text;
  finally
    lines.Free;
  end;

  jsonObj := TJSONObject.ParseJSONValue(jsonText) as TJSONObject;
  try
    if jsonObj = nil then
      raise Exception.Create('Failed to parse hparams.json');

    Result.ContextSize := jsonObj.GetValue<Integer>('n_ctx', 1024);
    Result.HeadCount := jsonObj.GetValue<Integer>('n_head', 12);
    Result.VocabularySize := jsonObj.GetValue<Integer>('n_vocab', 50257);
    Result.EmbeddingSize := jsonObj.GetValue<Integer>('n_embd', 768);
    Result.LayerCount := jsonObj.GetValue<Integer>('n_layer', 12);
  finally
    jsonObj.Free;
  end;
end;

end.
