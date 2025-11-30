unit MatrixUtility;

interface

uses
  System.SysUtils, System.Math;

type
  TMatrix1D = array of Single;
  TMatrix2D = array of array of Single;

  { --- Create arrays --- }

function CreateMatrix2D(rows, cols: Integer): TMatrix2D;
function CreateMatrix1D(rows: Integer): TMatrix1D;

{ --- Function / procedures --- }

function Add(const A: TMatrix2D; scalar: Single): TMatrix2D;
function AddRow(const A: TMatrix2D; const B: TMatrix1D): TMatrix2D;
function Mean(const A: TMatrix2D): Single;
function Multiply(const A: TMatrix2D; scalar: Single): TMatrix2D; overload;
function Multiply(const A: TMatrix1D; scalar: Single): TMatrix1D; overload;
function MultiplyDot(const A, B: TMatrix2D): TMatrix2D;
function MultiplyElementwise(const A, B: TMatrix2D): TMatrix2D; overload;
function MultiplyElementwise(const A: TMatrix1D; const B: TMatrix2D)
  : TMatrix2D; overload;
procedure PermuteInPlace(var A: TMatrix2D; seed: Integer);
function PowerMatrix(const A: TMatrix2D; p: Integer): TMatrix2D;
procedure RandomInPlace(const A: TMatrix2D; seed: Integer);
function Sigmoid(const A: TMatrix2D): TMatrix2D;
function SigmoidDerivative(const A: TMatrix2D): TMatrix2D;
procedure SplitRowsByRatio(const A: TMatrix2D; ratio: Single;
  out Set1, Set2: TMatrix2D);
procedure Standardize(const A: TMatrix2D; firstColIncl, lastColExcl: Integer);
function Subtract(const A, B: TMatrix2D): TMatrix2D; overload;
function Subtract(const A, B: TMatrix1D): TMatrix1D; overload;
function Sum(const A: TMatrix2D): Single;
function SumByColumn(const A: TMatrix2D): TMatrix1D;
function Transpose(const A: TMatrix2D): TMatrix2D;

implementation

uses
  System.Generics.Collections;

{ --- Create arrays --- }

function CreateMatrix2D(rows, cols: Integer): TMatrix2D;
var
  rowIndex, rowSize: Integer;
begin
  SetLength(Result, rows, cols);
  {rowSize := cols * SizeOf(Single);
  for rowIndex := 0 to rows - 1 do
    FillChar(Result[rowIndex][0], rowSize, 0);}
end;

function CreateMatrix1D(rows: Integer): TMatrix1D;
var
  rowSize: Integer;
begin
  SetLength(Result, rows);
  {rowSize := rows * SizeOf(Single);
  FillChar(Result, rowSize, 0);}
end;

{ --- Function / procedures --- }

function Add(const A: TMatrix2D; scalar: Single): TMatrix2D;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix2D(aRows, aCols);

  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i, j] := A[i, j] + scalar;
end;

function AddRow(const A: TMatrix2D; const B: TMatrix1D): TMatrix2D;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix2D(aRows, aCols);

  {if Length(B) <> aCols then
    raise Exception.Create(Format('Matrix column count mismatch %5d %5d', [Length(B), aCols]));}

  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i, j] := A[i, j] + B[j];
end;

function Mean(const A: TMatrix2D): Single;
var
  i, j, aRows, aCols: Integer;
  Sum: Single;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Sum := 0;
  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Sum := Sum + A[i][j];
  Result := Sum / (aRows * aCols);
end;

function Multiply(const A: TMatrix2D; scalar: Single): TMatrix2D; overload;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix2D(aRows, aCols);

  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i][j] := A[i][j] * scalar;
end;

function Multiply(const A: TMatrix1D; scalar: Single): TMatrix1D; overload;
var
  i, len: Integer;
begin
  len := Length(A);
  Result := CreateMatrix1D(len);

  for i := 0 to len - 1 do
    Result[i] := A[i] * scalar;
end;

function MultiplyElementwise(const A, B: TMatrix2D): TMatrix2D; overload;
var
  i, j, aRows, aCols, bRows, bCols, maxCols, maxRows: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  bRows := Length(B);
  bCols := Length(B[0]);
  maxCols := Max(aCols, bCols);
  maxRows := Max(aRows, bRows);
  Result := CreateMatrix2D(maxRows, maxCols);

  for i := 0 to maxRows - 1 do
    for j := 0 to maxCols - 1 do
      Result[i, j] := A[i mod aRows, j mod aCols] * B[i mod bRows, j mod bCols];
end;

function MultiplyElementwise(const A: TMatrix1D; const B: TMatrix2D)
  : TMatrix2D; overload;
var
  i, j, aCols, bRows, bCols, maxCols: Integer;
begin
  aCols := Length(A);
  bRows := Length(B);
  bCols := Length(B[0]);
  maxCols := Max(aCols, bCols);
  Result := CreateMatrix2D(bRows, maxCols);

  for i := 0 to bRows - 1 do
    for j := 0 to maxCols - 1 do
      Result[i, j] := A[j mod aCols] * B[i mod bRows, j mod bCols];
end;

function MultiplyDot(const A, B: TMatrix2D): TMatrix2D;
var
  i, j, k, aRows, aCols, bCols: Integer;
  Sum: Single;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  bCols := Length(B[0]);
  Result := CreateMatrix2D(aRows, bCols);

  for i := 0 to aRows - 1 do
    for j := 0 to bCols - 1 do
    begin
      Sum := 0;
      for k := 0 to aCols - 1 do
        Sum := Sum + A[i][k] * B[k][j];
      Result[i][j] := Sum;
    end;
end;

procedure PermuteInPlace(var A: TMatrix2D; seed: Integer);
var
  aRows, aCols: Integer;
  i, j, Col: Integer;
  Temp: Single;
begin
  RandSeed := seed;

  aRows := Length(A);

  aCols := Length(A[0]);

  // Fisher–Yates shuffle on rows
  for i := aRows - 1 downto 1 do
  begin
    j := Random(i + 1); // range 0..i

    if i <> j then
    begin
      // swap entire rows i and j
      for Col := 0 to aCols - 1 do
      begin
        Temp := A[i][Col];
        A[i][Col] := A[j][Col];
        A[j][Col] := Temp;
      end;
    end;
  end;
end;

function PowerMatrix(const A: TMatrix2D; p: Integer): TMatrix2D;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix2D(aRows, aCols);

  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i][j] := Power(A[i][j], p);
end;

procedure RandomInPlace(const A: TMatrix2D; seed: Integer);
var
  i, j, aRows, aCols: Integer;
begin
  RandSeed := seed;
  aRows := Length(A);
  aCols := Length(A[0]);
  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      A[i, j] := Random - 0.5; // Random returns 0..1
end;

function Sigmoid(const A: TMatrix2D): TMatrix2D;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix2D(aRows, aCols);

  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i, j] := 1 / (1 + Exp(-A[i, j]));
end;

function SigmoidDerivative(const A: TMatrix2D): TMatrix2D;
var
  i, j, aRows, aCols: Integer;
  Sigmoid: Single;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  SetLength(Result, aRows, aCols);
  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
    begin
      Sigmoid := 1 / (1 + Exp(-A[i, j]));
      Result[i, j] := Sigmoid * (1 - Sigmoid);
    end;
end;

procedure SplitRowsByRatio(const A: TMatrix2D; ratio: Single;
  out Set1, Set2: TMatrix2D);
var
  i, j, rows, cols, splitIdx: Integer;
begin
  rows := Length(A);
  cols := Length(A[0]);
  splitIdx := Trunc(rows * ratio);
  SetLength(Set1, splitIdx, cols);
  SetLength(Set2, rows - splitIdx, cols);
  for i := 0 to rows - 1 do
    for j := 0 to cols - 1 do
      if i < splitIdx then
        Set1[i, j] := A[i, j]
      else
        Set2[i - splitIdx, j] := A[i, j];
end;

procedure Standardize(const A: TMatrix2D; firstColIncl, lastColExcl: Integer);
var
  i, j, aRows, aCols: Integer;
  Sum, Mean, sumSq, stddev, value: Single;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  for j := firstColIncl to lastColExcl - 2 do
  begin
    Sum := 0;
    for i := 0 to aRows - 1 do
      Sum := Sum + A[i, j];
    Mean := Sum / aRows;
    sumSq := 0;
    for i := 0 to aRows - 1 do
    begin
      value := A[i, j] - Mean;
      sumSq := sumSq + value * value;
    end;
    stddev := Sqrt(sumSq / aRows);
    if stddev = 0 then
      stddev := 1;
    for i := 0 to aRows - 1 do
      A[i, j] := (A[i, j] - Mean) / stddev;
  end;
end;

function Subtract(const A, B: TMatrix2D): TMatrix2D; overload;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix2D(aRows, aCols);

  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[i][j] := A[i][j] - B[i][j];
end;

function Subtract(const A, B: TMatrix1D): TMatrix1D; overload;
var
  i, len: Integer;
begin
  len := Length(A);
  Result := CreateMatrix1D(len);

  for i := 0 to len - 1 do
    Result[i] := A[i] - B[i];
end;

function Sum(const A: TMatrix2D): Single;
var
  i, j, aRows, aCols: Integer;
begin
  Result := 0;
  aRows := Length(A);
  aCols := Length(A[0]);
  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result := Result + A[i, j];
end;

function SumByColumn(const A: TMatrix2D): TMatrix1D;
var
  i, j, aRows, aCols: Integer;
  Sum: Single;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix1D(aCols);

  for j := 0 to aCols - 1 do
  begin
    Sum := 0;
    for i := 0 to aRows - 1 do
      Sum := Sum + A[i, j];
    Result[j] := Sum;
  end;
end;

function Transpose(const A: TMatrix2D): TMatrix2D;
var
  i, j, aRows, aCols: Integer;
begin
  aRows := Length(A);
  aCols := Length(A[0]);
  Result := CreateMatrix2D(aCols, aRows);

  for i := 0 to aRows - 1 do
    for j := 0 to aCols - 1 do
      Result[j][i] := A[i][j];
end;

end.
