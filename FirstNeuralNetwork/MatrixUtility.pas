unit MatrixUtility;

interface

uses
  Math, SysUtils;

type
  TFloatArray = array of Single;
  TMatrix = array of array of Single;

{ --- Matrix utility functions --- }

function CreateMatrix(rows, cols: Integer): TMatrix;
function CreateFloatArray(rows: Integer): TFloatArray;
function MultiplyDot(const A, B: TMatrix): TMatrix;
function Subtract(const A, B: TMatrix): TMatrix;
function Transpose(const A: TMatrix): TMatrix;
function MultiplyScalar(const A: TMatrix; s: Single): TMatrix;
function PowerMatrix(const A: TMatrix; p: Integer): TMatrix;
function Mean(const A: TMatrix): Single;

implementation

{ --- Matrix utility functions --- }

function CreateMatrix(rows, cols: Integer): TMatrix;
var
  rowIndex, rowSize: Integer;
begin
  SetLength(Result, rows, cols);
  rowSize := cols * SizeOf(Single);
  for rowIndex := 0 to rows - 1 do
    FillChar(Result[rowIndex][0], rowSize, 0);
end;

function CreateFloatArray(rows: Integer): TFloatArray;
var
  rowSize: Integer;
begin
  SetLength(Result, rows);
  rowSize := rows * SizeOf(Single);
  FillChar(Result, rowSize, 0);
end;

function MultiplyDot(const A, B: TMatrix): TMatrix;
var
  i, j, k, aRows, aCols, bCols: Integer;
  sum: Single;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  bCols := Length(B[0]);
  Result := CreateMatrix(aRows, bCols);

  for i := 0 to aRows - 1 do
    for j := 0 to bCols - 1 do
    begin
      sum := 0;
      for k := 0 to aCols - 1 do
        sum := sum + A[i][k] * B[k][j];
      Result[i][j] := sum;
    end;
end;

function Subtract(const A, B: TMatrix): TMatrix;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix(aRows, aCols);

  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i][j] := A[i][j] - B[i][j];
end;

function Transpose(const A: TMatrix): TMatrix;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix(aCols, aRows);
  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[j][i] := A[i][j];
end;

function MultiplyScalar(const A: TMatrix; s: Single): TMatrix;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix(aRows, aCols);
  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i][j] := A[i][j] * s;
end;

function PowerMatrix(const A: TMatrix; p: Integer): TMatrix;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix(aRows, aCols);
  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i][j] := Power(A[i][j], p);
end;

procedure RandomInPlace(const A: TMatrix; Seed: Integer);
var
  Rnd: Random;
  Row, Col: Integer;
begin
  Rnd := TRandom.Create(Seed);
  try
    for Row := 0 to High(M) do
      for Col := 0 to High(M[Row]) do
        // Random number in [-0.5, 0.5]
        M[Row][Col] := Rnd.NextSingle - 0.5;
  finally
    Rnd.Free;
  end;
end;

function Mean(const A: TMatrix): Single;
var
  i, j, aRows, aCols: Integer;
  sum: Single;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  sum := 0;
  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      sum := sum + A[i][j];
  Result := sum / (aRows * aCols);
end;

end.
