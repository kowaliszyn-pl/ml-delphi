unit DataUtility;

interface

uses
  MatrixUtility;

procedure GetData(out ATrain, ATest: TMatrix2D; randomSeed: Integer;
  testSplitRatio: Single);

implementation

uses
  Classes,
  SysUtils;

function LoadCsv(filePath: string): TMatrix2D;
var
  lines: TStringList;
  i, j, rows, cols: Integer;
  value: string;
  values: TArray<string>;
  formatSettings: TFormatSettings;
begin
  lines := TStringList.Create;
  try
    lines.LoadFromFile(filePath);

    rows := lines.Count - 1;
    values := lines[1].Split([',']);
    cols := Length(values);

    Result := CreateMatrix2D(rows, cols);
    formatSettings.DecimalSeparator := '.';

    for i := 1 to lines.Count - 1 do
    begin
      values := lines[i].Split([',']);
      for j := 0 to cols - 1 do
      begin
        value := values[j].Trim(['"']);
        Result[i - 1][j] := StrToFloat(value, formatSettings);
      end;
    end;

  finally
    lines.Free;
  end;
end;

procedure GetData(out ATrain, ATest: TMatrix2D; randomSeed: Integer;
  testSplitRatio: Single);
var
  BostonData: TMatrix2D;
  inputFeatureCount: Integer;
begin
  BostonData := LoadCsv('..\..\..\..\data\Boston\BostonHousing.csv');

  { Number of independent variables (last column is target) }
  inputFeatureCount := Length(BostonData[0]) - 1;

  { Standardize features except target }
  Standardize(BostonData, 0, inputFeatureCount);

  { Shuffle rows }
  PermuteInPlace(BostonData, randomSeed);

  { Split into Train and Test }
  SplitRowsByRatio(BostonData, testSplitRatio, ATrain, ATest);
end;

end.
