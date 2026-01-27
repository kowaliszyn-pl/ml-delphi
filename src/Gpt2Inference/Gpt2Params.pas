unit Gpt2Params;

interface

uses
  MatrixUtility, Gpt2HParams;

type
  TGpt2LinearParams = record
    Weights: TMatrix2D;
    Bias: TMatrix1D;
  end;

  TGpt2LayerNormParams = record
    Gamma: TMatrix1D;
    Beta: TMatrix1D;
  end;

  TGpt2MultiHeadAttentionParams = record
    Projection: TGpt2LinearParams;
    OutputProjection: TGpt2LinearParams;
  end;

  TGpt2MultiLayerPerceptron = record
    FullyConnected: TGpt2LinearParams;
    OutputProjection: TGpt2LinearParams;
  end;

  TGpt2Block = record
    LayerNorm1: TGpt2LayerNormParams;
    Attention: TGpt2MultiHeadAttentionParams;
    LayerNorm2: TGpt2LayerNormParams;
    MultiLayerPerceptron: TGpt2MultiLayerPerceptron;
  end;

  TGpt2BlockArray = array of TGpt2Block;

  TGpt2Params = record
    TokenEmbeddings: TMatrix2D;
    PositionalEmbeddings: TMatrix2D;
    Blocks: TGpt2BlockArray;
    FinalLayerNorm: TGpt2LayerNormParams;
  end;

function CreateRandomGpt2Params(const hParams: TGpt2HParams; seed: Integer): TGpt2Params;
function LoadGpt2ParamsFromDirectory(const modelDirectory: string;
  const hParams: TGpt2HParams): TGpt2Params;

implementation

uses
  SysUtils, Classes, System.Generics.Collections;

const
  MAGIC_HEADER = 'GPT2WEIGHTS';
  FORMAT_VERSION = 1;

type
  TTensor = record
    Name: string;
    Shape: array of Integer;
    Data: TMatrix1D;
  end;

function CreateRandomNormal2D(rows, cols: Integer; seed: Integer): TMatrix2D;
var
  i, j: Integer;
begin
  RandSeed := seed;
  Result := CreateMatrix2D(rows, cols);
  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do
      Result[i, j] := Random - 0.5;
end;

function CreateRandomNormal1D(size: Integer; seed: Integer): TMatrix1D;
var
  i: Integer;
begin
  RandSeed := seed;
  Result := CreateMatrix1D(size);
  for i := 0 to size - 1 do
    Result[i] := Random - 0.5;
end;

function TensorAs2D(const tensor: TTensor): TMatrix2D;
var
  rows, cols, i, j, idx: Integer;
begin
  rows := tensor.Shape[0];
  cols := tensor.Shape[1];
  Result := CreateMatrix2D(rows, cols);
  idx := 0;
  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do
    begin
      Result[i, j] := tensor.Data[idx];
      Inc(idx);
    end;
end;

function CreateRandomGpt2Params(const hParams: TGpt2HParams; seed: Integer): TGpt2Params;
var
  i: Integer;
  blockSeed: Integer;
begin
  Result.TokenEmbeddings := CreateRandomNormal2D(hParams.VocabularySize,
    hParams.EmbeddingSize, seed);
  Result.PositionalEmbeddings := CreateRandomNormal2D(hParams.ContextSize,
    hParams.EmbeddingSize, seed + 1);

  SetLength(Result.Blocks, hParams.LayerCount);
  for i := 0 to hParams.LayerCount - 1 do
  begin
    blockSeed := seed + 100 + i * 10;

    Result.Blocks[i].LayerNorm1.Gamma := CreateRandomNormal1D(hParams.EmbeddingSize, blockSeed);
    Result.Blocks[i].LayerNorm1.Beta := CreateRandomNormal1D(hParams.EmbeddingSize, blockSeed + 1);

    Result.Blocks[i].Attention.Projection.Weights := CreateRandomNormal2D(
      hParams.EmbeddingSize, 3 * hParams.EmbeddingSize, blockSeed + 2);
    Result.Blocks[i].Attention.Projection.Bias := CreateRandomNormal1D(
      3 * hParams.EmbeddingSize, blockSeed + 3);
    Result.Blocks[i].Attention.OutputProjection.Weights := CreateRandomNormal2D(
      hParams.EmbeddingSize, hParams.EmbeddingSize, blockSeed + 4);
    Result.Blocks[i].Attention.OutputProjection.Bias := CreateRandomNormal1D(
      hParams.EmbeddingSize, blockSeed + 5);

    Result.Blocks[i].LayerNorm2.Gamma := CreateRandomNormal1D(hParams.EmbeddingSize, blockSeed + 6);
    Result.Blocks[i].LayerNorm2.Beta := CreateRandomNormal1D(hParams.EmbeddingSize, blockSeed + 7);

    Result.Blocks[i].MultiLayerPerceptron.FullyConnected.Weights := CreateRandomNormal2D(
      hParams.EmbeddingSize, 4 * hParams.EmbeddingSize, blockSeed + 8);
    Result.Blocks[i].MultiLayerPerceptron.FullyConnected.Bias := CreateRandomNormal1D(
      4 * hParams.EmbeddingSize, blockSeed + 9);
    Result.Blocks[i].MultiLayerPerceptron.OutputProjection.Weights := CreateRandomNormal2D(
      4 * hParams.EmbeddingSize, hParams.EmbeddingSize, blockSeed + 10);
    Result.Blocks[i].MultiLayerPerceptron.OutputProjection.Bias := CreateRandomNormal1D(
      hParams.EmbeddingSize, blockSeed + 11);
  end;

  Result.FinalLayerNorm.Gamma := CreateRandomNormal1D(hParams.EmbeddingSize, seed + 2);
  Result.FinalLayerNorm.Beta := CreateRandomNormal1D(hParams.EmbeddingSize, seed + 3);
end;

function LoadGpt2ParamsFromDirectory(const modelDirectory: string;
  const hParams: TGpt2HParams): TGpt2Params;
var
  path: string;
  stream: TFileStream;
  reader: TBinaryReader;
  magic: TBytes;
  version, tensorCount, i, j, rank, dimension: Integer;
  nameLen: Integer;
  nameBytes: TBytes;
  tensorName, prefix: string;
  tensor: TTensor;
  elementCount: Int64;
  tensors: TDictionary<string, TTensor>;
begin
  path := IncludeTrailingPathDelimiter(modelDirectory) + 'weights.bin';
  stream := TFileStream.Create(path, fmOpenRead or fmShareDenyWrite);
  tensors := TDictionary<string, TTensor>.Create;
  try
    reader := TBinaryReader.Create(stream);
    try
      { Read and verify magic header }
      SetLength(magic, Length(MAGIC_HEADER));
      reader.Read(magic, 0, Length(MAGIC_HEADER));
      if TEncoding.ASCII.GetString(magic) <> MAGIC_HEADER then
        raise Exception.Create('Unsupported GPT-2 weight file (invalid magic header).');

      { Read and verify version }
      version := reader.ReadInt32;
      if version <> FORMAT_VERSION then
        raise Exception.CreateFmt('Unsupported GPT-2 weight file version %d.', [version]);

      { Read tensor count }
      tensorCount := reader.ReadInt32;

      { Read all tensors }
      for i := 0 to tensorCount - 1 do
      begin
        { Read tensor name }
        nameLen := reader.ReadInt32;
        SetLength(nameBytes, nameLen);
        reader.Read(nameBytes, 0, nameLen);
        tensorName := TEncoding.UTF8.GetString(nameBytes);

        { Read rank and shape }
        rank := reader.ReadInt32;
        SetLength(tensor.Shape, rank);
        elementCount := 1;
        for j := 0 to rank - 1 do
        begin
          dimension := reader.ReadInt32;
          tensor.Shape[j] := dimension;
          elementCount := elementCount * dimension;
        end;

        { Read data }
        SetLength(tensor.Data, elementCount);
        for j := 0 to elementCount - 1 do
          tensor.Data[j] := reader.ReadSingle;

        tensor.Name := tensorName;
        tensors.Add(tensorName, tensor);
      end;
    finally
      reader.Free;
    end;

    { Build Gpt2Params from tensors }
    Result.TokenEmbeddings := TensorAs2D(tensors['token_embeddings']);
    Result.PositionalEmbeddings := TensorAs2D(tensors['positional_embeddings']);
    Result.FinalLayerNorm.Gamma := tensors['final_layer_norm.gamma'].Data;
    Result.FinalLayerNorm.Beta := tensors['final_layer_norm.beta'].Data;

    SetLength(Result.Blocks, hParams.LayerCount);
    for i := 0 to hParams.LayerCount - 1 do
    begin
      prefix := Format('blocks.%d.', [i]);

      Result.Blocks[i].LayerNorm1.Gamma := tensors[prefix + 'ln1.gamma'].Data;
      Result.Blocks[i].LayerNorm1.Beta := tensors[prefix + 'ln1.beta'].Data;

      Result.Blocks[i].Attention.Projection.Weights := TensorAs2D(tensors[prefix + 'attn.qkv.weight']);
      Result.Blocks[i].Attention.Projection.Bias := tensors[prefix + 'attn.qkv.bias'].Data;
      Result.Blocks[i].Attention.OutputProjection.Weights := TensorAs2D(tensors[prefix + 'attn.out.weight']);
      Result.Blocks[i].Attention.OutputProjection.Bias := tensors[prefix + 'attn.out.bias'].Data;

      Result.Blocks[i].LayerNorm2.Gamma := tensors[prefix + 'ln2.gamma'].Data;
      Result.Blocks[i].LayerNorm2.Beta := tensors[prefix + 'ln2.beta'].Data;

      Result.Blocks[i].MultiLayerPerceptron.FullyConnected.Weights := TensorAs2D(tensors[prefix + 'mlp.up.weight']);
      Result.Blocks[i].MultiLayerPerceptron.FullyConnected.Bias := tensors[prefix + 'mlp.up.bias'].Data;
      Result.Blocks[i].MultiLayerPerceptron.OutputProjection.Weights := TensorAs2D(tensors[prefix + 'mlp.down.weight']);
      Result.Blocks[i].MultiLayerPerceptron.OutputProjection.Bias := tensors[prefix + 'mlp.down.bias'].Data;
    end;
  finally
    tensors.Free;
    stream.Free;
  end;
end;

end.
