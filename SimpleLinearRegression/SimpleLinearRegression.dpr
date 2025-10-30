program SimpleLinearRegression;

{$APPTYPE CONSOLE}

uses
  SysUtils, Windows;

const
  { 1. Set the parameters for the model }
  LearningRate: Single = 0.0005;
  Iterations: Integer = 35000;
  PrintEvery: Integer = 1000;

type
  TSample = array[0..1] of Single;
  TDataset = array[0..4] of TSample;

var
  data: TDataset;
  a, b: Single;
  i, j, n: Integer;
  error, squaredError, sumErrorValue, sumError, meanSquaredError: Single;
  x, y, prediction: Single;
  deltaA, deltaB: Single;

begin
  SetConsoleOutputCP(CP_UTF8);

  { 2. Prepare training data }
  data[0][0] := 10; data[0][1] := 100;
  data[1][0] := 20; data[1][1] := 80;
  data[2][0] := 30; data[2][1] := 60;
  data[3][0] := 40; data[3][1] := 40;
  data[4][0] := 50; data[4][1] := 20;

  { 3. Initialize model }
  a := 0;
  b := 0;

  { Number of samples }
  n := Length(data);

  { 4. Training loop }
  for i := 0 to Iterations - 1 do
  begin
    { Initialize accumulators for errors }
    sumErrorValue := 0;
    sumError := 0;
    squaredError := 0;

    for j := 0 to n - 1 do
    begin
      x := data[j][0];
      y := data[j][1];

      { Prediction and error calculation }
      prediction := a * x + b;
      error := y - prediction;

      { Accumulate squared error and gradients }
      squaredError := squaredError + error * error;
      sumErrorValue := sumErrorValue + error * x;
      sumError := sumError + error;
    end;

    meanSquaredError := squaredError / n;
    deltaA := -2.0 / n * sumErrorValue;
    deltaB := -2.0 / n * sumError;

    { Update regression parameters }
    a := a - LearningRate * deltaA;
    b := b - LearningRate * deltaB;

    if (i mod PrintEvery = 0) then
      Writeln(Format('Iteration: %5d | MSE: %10.5f | ∂MSE/∂a: %10.4f | ∂MSE/∂b: %10.4f | a: %9.4f | b: %9.4f',
  [i, meanSquaredError, deltaA, deltaB, a, b]));
  end;

  { 5. Output learned parameters }
  Writeln;
  Writeln(Format('%-20s a = %9.4f | b = %9.4f', ['Learned parameters:', a, b]));
  Writeln(Format('%-20s a = %9.4f | b = %9.4f', ['Expected parameters:', -2.0, 120.0]));
  Readln;
end.

