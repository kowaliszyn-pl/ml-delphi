unit DataUtility;

interface

uses
  MatrixUtility;

procedure GetData(out ATrain, ATest: TMatrix2D; const RandomSeed: Integer; const TestSplitRatio: Single);

implementation

function LoadCsv(const FilePath: string): TMatrix2D;
var
  Lines: TStringList;
  I, J, Rows, Cols: Integer;
  Values: TArray<string>;
begin
  Lines := TStringList.Create;
  try
    Lines.LoadFromFile(FilePath);

    // Skip header line
    if Lines.Count = 0 then
      Exit(nil);

    Rows := Lines.Count - 1;
    Values := Lines[1].Split([',']);
    Cols := Length(Values);

    SetLength(Result, Rows, Cols);

    for I := 1 to Lines.Count - 1 do
    begin
      Values := Lines[I].Split([',']);
      for J := 0 to Cols - 1 do
        Result[I-1][J] := StrToFloat(StringReplace(Values[J].Trim(['"']),
                            ',', '.', [rfReplaceAll]));
    end;

  finally
    Lines.Free;
  end;
end;

procedure GetData(out ATrain, ATest: TMatrix2D; const RandomSeed: Integer; const TestSplitRatio: Single);
var
  BostonData: TMatrix2D;
  InputFeatureCount: Integer;
begin
  BostonData := LoadCsv('..\..\..\..\..\data\Boston\BostonHousing.csv');

  { Number of independent variables (last column is target) }
  InputFeatureCount := Length(BostonData[0]) - 2; // last index = cols-1, so end at cols-2

  { Standardize features except target }
  Standardize(BostonData, 0, InputFeatureCount);

  { Shuffle rows }
  PermuteInPlace(BostonData, RandomSeed);

  { Return (Train, Test) }
  Result := SplitRowsByRatio(BostonData, TestSplitRatio);
end;

end.
