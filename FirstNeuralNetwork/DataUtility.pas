unit DataUtility;

interface

uses
  MatrixUtility;

procedure GetData(out ATrain, ATest: TMatrix2D; const RandomSeed: Integer;
  const TestSplitRatio: Single);

implementation

uses
  Classes, SysUtils;

function LoadCsv(const FilePath: string): TMatrix2D;
var
  Lines: TStringList;
  I, J, Rows, Cols: Integer;
  value: String;
  Values: TArray<String>;
  MyFormatSettings: TFormatSettings;
begin
  Lines := TStringList.Create;
  try
    Lines.LoadFromFile(FilePath);

    Rows := Lines.Count - 1;
    Values := Lines[1].Split([',']);
    Cols := Length(Values);

    Result := CreateMatrix2D(Rows, Cols);
    MyFormatSettings.DecimalSeparator := '.';

    for I := 1 to Lines.Count - 1 do
    begin
      Values := Lines[I].Split([',']);
      for J := 0 to Cols - 1 do
      begin
        value := Values[J].Trim(['"']);
        // value := StringReplace(value, ',', '.', [rfReplaceAll]);
        Result[I - 1][J] := StrToFloat(value, MyFormatSettings);
      end;
    end;

  finally
    Lines.Free;
  end;
end;

procedure GetData(out ATrain, ATest: TMatrix2D; const RandomSeed: Integer;
  const TestSplitRatio: Single);
var
  BostonData: TMatrix2D;
  InputFeatureCount: Integer;
begin
  BostonData := LoadCsv('..\..\..\data\Boston\BostonHousing.csv');

  { Number of independent variables (last column is target) }
  InputFeatureCount := Length(BostonData[0]) - 2;
  // last index = cols-1, so end at cols-2

  { Standardize features except target }
  Standardize(BostonData, 0, InputFeatureCount);

  { Shuffle rows }
  PermuteInPlace(BostonData, RandomSeed);

  { Split into Train and Test }
  SplitRowsByRatio(BostonData, TestSplitRatio, ATrain, ATest);
end;

end.
