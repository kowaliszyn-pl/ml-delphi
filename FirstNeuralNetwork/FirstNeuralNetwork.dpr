program FirstNeuralNetwork;

{$APPTYPE CONSOLE}

uses
  SysUtils,
  Windows,
  DataUtility in 'DataUtility.pas',
  MatrixUtility in 'MatrixUtility.pas';

const
  { Hyperparameters for the model }
  LearningRate: Single = 0.0005;
  Iterations: Integer = 48000;
  PrintEvery: Integer = 2000;
  TestSplitRatio: Single = 0.7;
  RandomSeed: Integer = 251113;
  HiddenLayerSize: Integer = 4;

var
  trainData, testData: TMatrix2D;
  XTrain, YTrain, XTest, YTest: TMatrix2D;
  W1, W2, XTrainT, M1, N1, O1, M2, predictions, errors: TMatrix2D;
  M1Test, N1Test, O1Test, M2Test, testPredictions, testErrors: TMatrix2D;
  dLdP, dLdW2, dLdO1, dLdN1, dLdW1: TMatrix2D;
  B1, dLdBias1: TMatrix1D;
  b2, dLdBias2, negativeTwoOverN, meanSquaredError: Single;
  i, j, iteration, inputFeatureCount, nTrain, nTest: Integer;
  showTestSamples: array of Integer;

begin
  SetConsoleOutputCP(CP_UTF8);

  { Load data (trainData, testData) }
  GetData(trainData, testData, RandomSeed, TestSplitRatio);

  { Prepare XTrain, YTrain, XTest, YTest }
  inputFeatureCount := Length(trainData[0]) - 1;
  nTrain := Length(trainData);
  nTest := Length(testData);

  XTrain := CreateMatrix2D(nTrain, inputFeatureCount);
  YTrain := CreateMatrix2D(nTrain, 1);

  XTest := CreateMatrix2D(nTest, inputFeatureCount);
  YTest := CreateMatrix2D(nTest, 1);

  { Fill XTrain / YTrain }
  for i := 0 to nTrain - 1 do
  begin
    for j := 0 to inputFeatureCount - 1 do
      XTrain[i][j] := trainData[i][j];
    YTrain[i][0] := trainData[i][inputFeatureCount];
  end;

  { Fill XTest / YTest }
  for i := 0 to nTest - 1 do
  begin
    for j := 0 to inputFeatureCount - 1 do
      XTest[i][j] := testData[i][j];
    YTest[i][0] := testData[i][inputFeatureCount];
  end;

  { Initialize parameters: W1, B1, W2, b2 }
  W1 := CreateMatrix2D(inputFeatureCount, HiddenLayerSize);
  RandomInPlace(W1, RandomSeed);
  B1 := CreateMatrix1D(HiddenLayerSize);

  W2 := CreateMatrix2D(HiddenLayerSize, 1);
  { We use RandomSeed + 1 because we want different random values than for W1 }
  RandomInPlace(W2, RandomSeed + 1);
  b2 := 0.0;

  { Precompute common quantities }
  XTrainT := Transpose(XTrain);
  negativeTwoOverN := -2.0 / nTrain;

  { Training loop (forward + backward) }
  for iteration := 1 to Iterations do
  begin
    { Forward pass }
    M1 := MultiplyDot(XTrain, W1);
    N1 := AddRow(M1, B1);
    O1 := Sigmoid(N1);

    M2 := MultiplyDot(O1, W2);
    predictions := Add(M2, b2);

    errors := Subtract(YTrain, predictions);

    { Backward pass }

    { The second layer (output) }
    dLdP := Multiply(errors, negativeTwoOverN);
    dLdW2 := MultiplyDot(Transpose(O1), dLdP);
    dLdBias2 := Sum(dLdP);

    { The first layer (hidden) }
    dLdO1 := MultiplyDot(dLdP, Transpose(W2));
    dLdN1 := MultiplyElementwise(dLdO1, SigmoidDerivative(N1)); // element-wise
    dLdBias1 := SumByColumn(dLdN1);
    dLdW1 := MultiplyDot(XTrainT, dLdN1);

    { Update parameters }
    W1 := Subtract(W1, Multiply(dLdW1, LearningRate));
    W2 := Subtract(W2, Multiply(dLdW2, LearningRate));
    B1 := Subtract(B1, Multiply(dLdBias1 * LearningRate);
    b2 := b2 - (dLdBias2 * LearningRate);

    if (iteration mod PrintEvery) = 0 then
    begin
      { Mean Squared Error }
      meanSquaredError := Mean(PowerMatrix(errors, 2));

      Writeln(Format('Iteration: %6d | MSE: %8.5f',
        [iteration, meanSquaredError]));
    end;

    { Free intermediate matrices if your MatrixUtility requires manual freeing.
      If the utility uses dynamic arrays and automatic memory, you can omit. }
  end;

  { Print learned parameters (W1, B1, W2, b2) }
  Writeln;
  Writeln('--- Training Complete (Simplified Neural Network) ---');
  Writeln('Learned parameters:');
  Writeln('Weights for the first layer (W1):');
  for i := 0 to Length(W1) - 1 do
  begin
    for j := 0 to Length(W1[0]) - 1 do
      Write(Format('%8.4f ', [W1[i][j]]));
    Writeln;
  end;

  Writeln('Biases for the first layer (B1):');
  for j := 0 to High(B1) do
    Writeln(Format(' B1[%d] = %8.4f', [j, B1[j]]));

  Writeln('Weights for the second layer (W2):');
  for i := 0 to Length(W2) - 1 do
  begin
    for j := 0 to Length(W2[0]) - 1 do
      Write(Format('%8.4f ', [W2[i][j]]));
    Writeln;
  end;

  Writeln(Format('Bias for the second layer (b2): %8.4f', [b2]));
  Writeln;

  { Evaluate on test set: forward pass for test samples }
  M1Test := MultiplyDot(XTest, W1);
  N1Test := AddRow(M1Test, B1);
  O1Test := Sigmoid(N1Test);
  M2Test := MultiplyDot(O1Test, W2);
  testPredictions := Add(M2Test, b2);

  Writeln('Sample predictions vs actual values:');
  Writeln(Format('%14s%14s%14s', ['Sample No', 'Predicted', 'Actual']));

  SetLength(showTestSamples, 6);
  showTestSamples[0] := 0;
  showTestSamples[1] := 1;
  showTestSamples[2] := 2;
  showTestSamples[3] := nTest - 3;
  showTestSamples[4] := nTest - 2;
  showTestSamples[5] := nTest - 1;

  for i := 0 to High(showTestSamples) do
  begin
    if (showTestSamples[i] < 0) or (showTestSamples[i] >= nTest) then
      Continue;
    Writeln(Format('%14d%14.4f%14.4f', [showTestSamples[i] + 1,
      testPredictions[showTestSamples[i]][0], YTest[showTestSamples[i]][0]]));
  end;

  testErrors := Subtract(YTest, testPredictions);
  meanSquaredError := Mean(PowerMatrix(testErrors, 2));
  // highlight output - simple approach: print with stars
  Writeln;
  Writeln(Format('MSE on test data: %8.5f', [meanSquaredError]));

  Readln;

end.
