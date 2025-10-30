unit MatrixUtility;

interface

uses
  Math;

type
  TMatrix = array of array of Single;

{ --- Matrix utility functions --- }

function CreateMatrix(rows, cols: Integer): TMatrix;
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
  r: Integer;
begin
  SetLength(Result, rows, cols);
  for r := 0 to rows - 1 do
    FillChar(Result[r][0], cols * SizeOf(Single), 0);
end;

function MultiplyDot(const A, B: TMatrix): TMatrix;
var
  i, j, k, aRows, aCols, bCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  bCols := Length(B[0]);
  Result := CreateMatrix(aRows, bCols);

  for i := 0 to aRows - 1 do
    for j := 0 to bCols - 1 do
    begin
      Result[i][j] := 0;
      for k := 0 to aCols - 1 do
        Result[i][j] := Result[i][j] + A[i][k] * B[k][j];
    end;
end;

function Subtract(const A, B: TMatrix): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do
      Result[i][j] := A[i][j] - B[i][j];
end;

function Transpose(const A: TMatrix): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A[0]), Length(A));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do
      Result[j][i] := A[i][j];
end;

function MultiplyScalar(const A: TMatrix; s: Single): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do
      Result[i][j] := A[i][j] * s;
end;

function PowerMatrix(const A: TMatrix; p: Integer): TMatrix;
var
  i, j: Integer;
begin
  Result := CreateMatrix(Length(A), Length(A[0]));
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do
      Result[i][j] := Power(A[i][j], p);
end;

function Mean(const A: TMatrix): Single;
var
  i, j, count: Integer;
  sum: Single;
begin
  sum := 0;
  count := 0;
  for i := 0 to High(A) do
    for j := 0 to High(A[0]) do
    begin
      sum := sum + A[i][j];
      Inc(count);
    end;
  Result := sum / count;
end;

end.
