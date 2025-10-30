program MultiLinearRegression;

{$APPTYPE CONSOLE}

uses
  SysUtils,
  Windows,
  Math,
  MatrixUtility in 'MatrixUtility.pas';

const
  { 1. Set the parameters for the model }
  LearningRate: Single = 0.0005;
  Iterations: Integer = 35000;
  PrintEvery: Integer = 1000;

type
  TSample = array[0..3] of Single; // [x1, x2, x3, y]
  TDataset = array[0..4] of TSample;


var
  XAnd1, Y, AB, Predictions, Errors, DeltaAB: TMatrix;
  i, j, k, n, numCoefficients: Integer;
  meanSquaredError, tempSum: Single;

const
  { y = 2*x1 + 3*x2 - 1*x3 + 5 }
  data: TDataset = (
    (1, 2, 1, 12),
    (2, 1, 2, 10),
    (3, 3, 1, 19),
    (4, 2, 3, 16),
    (1, 4, 2, 17)
  );

begin
  SetConsoleOutputCP(CP_UTF8);

  { Number of samples and coefficients }
  n := Length(data);
  numCoefficients := Length(data[0]) - 1;

  { Feature matrix XAnd1 with bias term, and target vector Y }
  XAnd1 := CreateMatrix(n, numCoefficients + 1);
  Y := CreateMatrix(n, 1);

  for i := 0 to n - 1 do
  begin
    for j := 0 to numCoefficients - 1 do
      XAnd1[i][j] := data[i][j];
    XAnd1[i][numCoefficients] := 1; // Bias term
    Y[i][0] := data[i][numCoefficients];
  end;

  { 3. Initialize parameters (a1, a2, a3, b) }
  AB := CreateMatrix(numCoefficients + 1, 1);

  { 4. Training loop }
  for i := 1 to Iterations do
  begin
    { Predictions and errors }
    Predictions := MultiplyDot(XAnd1, AB);
    Errors := Subtract(Y, Predictions);

    { Mean Squared Error }
    meanSquaredError := Mean(PowerMatrix(Errors, 2));

    { Gradient calculation: ∂MSE/∂AB = -2/n * X^T * Errors }
    DeltaAB := MultiplyScalar(MultiplyDot(Transpose(XAnd1), Errors), -2.0 / n);

    { Update parameters }
    AB := Subtract(AB, MultiplyScalar(DeltaAB, LearningRate));

    if (i mod PrintEvery = 0) then
      Writeln(Format('Iteration: %6d | MSE: %8.5f | a1: %8.4f | a2: %8.4f | a3: %8.4f | b: %8.4f',
        [i, meanSquaredError, AB[0][0], AB[1][0], AB[2][0], AB[3][0]]));
  end;

  { 5. Output learned parameters }
  Writeln;
  Writeln('--- Training Complete (Matrices with Bias) ---');
  Writeln(Format('%-20s a1 = %9.4f | a2 = %9.4f | a3 = %9.4f | b = %9.4f',
    ['Learned parameters:', AB[0][0], AB[1][0], AB[2][0], AB[3][0]]));
  Writeln(Format('%-20s a1 = %9.4f | a2 = %9.4f | a3 = %9.4f | b = %9.4f',
    ['Expected parameters:', 2.0, 3.0, -1.0, 5.0]));

  Readln;
end.
