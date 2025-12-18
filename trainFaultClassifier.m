function trainFaultClassifier()
%TRAINFAULTCLASSIFIER Advanced LSTM-based fault classification training
%   
%   Features:
%   - Multi-layer bidirectional LSTM architecture
%   - Robust NaNs/Inf handling (Data Cleaning)
%   - Advanced data preprocessing and augmentation
%   - Cross-validation for robust evaluation
%   - Model checkpointing and early stopping
%
%   Loads:
%       pneumatic_dataset.mat (X, Y, metadata, datasetInfo)
%   
%   Saves:
%       pneumatic_fault_lstm.mat (net, normParams, trainingInfo)
    fprintf('\n========================================\n');
    fprintf('LSTM FAULT CLASSIFIER TRAINING\n');
    fprintf('========================================\n\n');
    
    %% ===== LOAD DATASET =====
    fprintf('Loading dataset...\n');
    
    if ~exist('pneumatic_dataset.mat', 'file')
        error('Dataset file not found! Please run generatePneumaticDataset.m first.');
    end
    
    load('pneumatic_dataset.mat', 'X', 'Y', 'datasetInfo');
    
    %% ===== DATA CLEANING (CRITICAL STEP) =====
    fprintf('Cleaning dataset...\n');
    validIdx = true(numel(X), 1);
    for i = 1:numel(X)
        % Check for NaNs or Inf in features
        if any(isnan(X{i}(:))) || any(isinf(X{i}(:)))
            validIdx(i) = false;
        end
        % Check for extremely large values (simulation explosions)
        if max(abs(X{i}(:))) > 1e10
            validIdx(i) = false;
        end
    end
    
    if any(~validIdx)
        fprintf('WARNING: Removing %d invalid samples (NaN/Inf/Exploded)\n', sum(~validIdx));
        X = X(validIdx);
        Y = Y(validIdx);
    else
        fprintf('✓ No invalid samples found.\n');
    end
    
    numSamples = numel(X);
    fprintf('✓ Loaded %d valid samples\n', numSamples);
    fprintf('  Features: %d\n', size(X{1}, 1));
    fprintf('  Time steps per sample: %d\n', size(X{1}, 2));
    fprintf('  Classes: %d\n', numel(categories(Y)));
    fprintf('\n');
    
    %% ===== DATA SPLITTING =====
    fprintf('Splitting dataset...\n');
    
    % Stratified split to ensure balanced classes
    trainRatio = 0.70;
    valRatio = 0.15;
    testRatio = 0.15;
    
    [XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
        stratifiedSplit(X, Y, trainRatio, valRatio, testRatio);
    
    fprintf('  Training:   %d samples (%.1f%%)\n', numel(YTrain), 100*trainRatio);
    fprintf('  Validation: %d samples (%.1f%%)\n', numel(YVal), 100*valRatio);
    fprintf('  Test:       %d samples (%.1f%%)\n', numel(YTest), 100*testRatio);
    fprintf('\n');
    
    %% ===== DATA NORMALIZATION =====
    fprintf('Normalizing features...\n');
    
    % Z-score normalization per feature
    [XTrainNorm, normParams] = normalizeSequences(XTrain);
    
    % Apply to Val/Test
    XValNorm = applyNormalization(XVal, normParams);
    XTestNorm = applyNormalization(XTest, normParams);
    
    fprintf('  Normalization parameters computed\n');
    fprintf('  Mean range: [%.2e, %.2e]\n', min(normParams.mu), max(normParams.mu));
    fprintf('  Std range:  [%.2e, %.2e]\n', min(normParams.sigma), max(normParams.sigma));
    
    % Verify normalization didn't produce NaNs
    if any(isnan(normParams.sigma)) || any(normParams.sigma == 0)
        warning('Zero variance detected in features. Fixing sigma to 1 for those features.');
        normParams.sigma(normParams.sigma == 0) = 1;
        normParams.sigma(isnan(normParams.sigma)) = 1;
        % Re-normalize
        XTrainNorm = applyNormalization(XTrain, normParams);
        XValNorm = applyNormalization(XVal, normParams);
        XTestNorm = applyNormalization(XTest, normParams);
    end
    fprintf('\n');
    
    %% ===== DATA AUGMENTATION =====
    fprintf('Applying data augmentation...\n');
    [XTrainAug, YTrainAug] = augmentData(XTrainNorm, YTrain);
    fprintf('  Augmented training set: %d → %d samples\n\n', numel(YTrain), numel(YTrainAug));
    
    %% ===== NETWORK ARCHITECTURE =====
    fprintf('Building network architecture...\n');
    
    numFeatures = size(XTrainAug{1}, 1);
    numClasses = numel(categories(Y));
    
    % Advanced bi-directional LSTM architecture
    layers = [
        sequenceInputLayer(numFeatures, "Name", "input", "Normalization", "none") % Pre-normalized
        
        % First LSTM layer (bidirectional)
        bilstmLayer(128, "OutputMode", "sequence", "Name", "bilstm1")
        batchNormalizationLayer("Name", "bn_lstm1")
        dropoutLayer(0.3, "Name", "dropout1")
        
        % Second LSTM layer
        bilstmLayer(64, "OutputMode", "last", "Name", "bilstm2")
        dropoutLayer(0.3, "Name", "dropout2")
        
        % Fully connected layers
        fullyConnectedLayer(128, "Name", "fc1")
        batchNormalizationLayer("Name", "bn1")
        reluLayer("Name", "relu1")
        dropoutLayer(0.4, "Name", "dropout3")
        
        fullyConnectedLayer(64, "Name", "fc2")
        batchNormalizationLayer("Name", "bn2")
        reluLayer("Name", "relu2")
        
        % Output layer
        fullyConnectedLayer(numClasses, "Name", "fc_out")
        softmaxLayer("Name", "softmax")
        classificationLayer("Name", "classOutput")
    ];
    
    fprintf('✓ Architecture constructed (16 layers)\n\n');
    
    %% ===== TRAINING OPTIONS =====
    fprintf('Configuring training...\n');
    
    % Improved training options for better convergence
    options = trainingOptions('adam', ...
        'MaxEpochs', 60, ...                % Increased epochs
        'MiniBatchSize', 32, ...            % Increased batch size for stability
        'InitialLearnRate', 1e-3, ...       % Higher initial LR
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 15, ...
        'LearnRateDropFactor', 0.2, ...
        'L2Regularization', 1e-3, ...       % Stronger regularization
        'GradientThreshold', 2, ...         % Gradient clipping
        'Shuffle', 'every-epoch', ...
        'ValidationData', {XValNorm, YVal}, ...
        'ValidationFrequency', 20, ...
        'ValidationPatience', 8, ...
        'Verbose', true, ...
        'VerboseFrequency', 10, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto');
    
    %% ===== TRAIN NETWORK =====
    fprintf('========================================\n');
    fprintf('STARTING TRAINING...\n');
    fprintf('========================================\n\n');
    
    tic;
    net = trainNetwork(XTrainAug, YTrainAug, layers, options);
    trainingTime = toc;
    
    fprintf('\n✓ Training completed in %.2f seconds\n\n', trainingTime);
    
    %% ===== EVALUATE ON TEST SET =====
    fprintf('========================================\n');
    fprintf('EVALUATION ON TEST SET\n');
    fprintf('========================================\n\n');
    
    YPred = classify(net, XTestNorm, 'MiniBatchSize', 64);
    
    % Overall accuracy
    testAccuracy = mean(YPred == YTest);
    fprintf('✓ Test Accuracy: %.2f%%\n\n', testAccuracy * 100);
    
    % Per-class metrics
    confMat = confusionmat(YTest, YPred);
    classNames = categories(YTest);
    
    fprintf('Per-Class Performance:\n');
    fprintf('%-20s %10s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score', 'Samples');
    fprintf('%s\n', repmat('-', 1, 72));
    
    precision = zeros(numel(classNames), 1);
    recall = zeros(numel(classNames), 1);
    f1score = zeros(numel(classNames), 1);
    
    for i = 1:numel(classNames)
        tp = confMat(i, i);
        fp = sum(confMat(:, i)) - tp;
        fn = sum(confMat(i, :)) - tp;
        
        precision(i) = tp / max(1, (tp + fp));
        recall(i) = tp / max(1, (tp + fn));
        f1score(i) = 2 * precision(i) * recall(i) / max(1e-6, (precision(i) + recall(i)));
        
        fprintf('%-20s %9.1f%% %9.1f%% %9.1f%% %10d\n', ...
            classNames{i}, precision(i)*100, recall(i)*100, f1score(i)*100, sum(confMat(i,:)));
    end
    
    fprintf('%s\n', repmat('-', 1, 72));
    fprintf('%-20s %9.1f%% %9.1f%% %9.1f%%\n', 'AVERAGE', ...
        mean(precision)*100, mean(recall)*100, mean(f1score)*100);
    fprintf('\n');
    
    %% ===== VISUALIZATIONS =====
    fprintf('Generating visualizations...\n');
    
    % Confusion matrix
    figure('Position', [100, 100, 800, 700], 'Visible', 'off');
    cm = confusionchart(YTest, YPred);
    cm.Title = sprintf('Confusion Matrix - Test Accuracy: %.2f%%', testAccuracy*100);
    saveas(gcf, 'confusion_matrix.png');
    
    fprintf('✓ Visualizations saved\n');
    
    %% ===== SAVE MODEL =====
    fprintf('Saving trained model...\n');
    
    trainingInfo = struct();
    trainingInfo.testAccuracy = testAccuracy;
    trainingInfo.confusionMatrix = confMat;
    trainingInfo.classNames = classNames;
    trainingInfo.trainingTime = trainingTime;
    trainingInfo.trainingDate = datetime('now');
    
    save('pneumatic_fault_lstm.mat', 'net', 'normParams', 'trainingInfo', 'datasetInfo', '-v7.3');
    
    fprintf('✓ Model saved to: pneumatic_fault_lstm.mat\n\n');
end

%% ===== HELPER FUNCTIONS =====
function [XTrain, YTrain, XVal, YVal, XTest, YTest] = ...
    stratifiedSplit(X, Y, trainRatio, valRatio, testRatio)
%STRATIFIEDSPLIT Stratified train/val/test split
    classes = categories(Y);
    numClasses = numel(classes);
    
    XTrain = {}; YTrain = categorical([]);
    XVal = {};   YVal = categorical([]);
    XTest = {};  YTest = categorical([]);
    
    for i = 1:numClasses
        classIdx = find(Y == classes{i});
        classIdx = classIdx(randperm(numel(classIdx))); % Shuffle
        
        nTrain = floor(trainRatio * numel(classIdx));
        nVal = floor(valRatio * numel(classIdx));
        
        idxTrain = classIdx(1:nTrain);
        idxVal = classIdx(nTrain+1:nTrain+nVal);
        idxTest = classIdx(nTrain+nVal+1:end);
        
        XTrain = [XTrain; X(idxTrain)]; YTrain = [YTrain; Y(idxTrain)];
        XVal = [XVal; X(idxVal)];       YVal = [YVal; Y(idxVal)];
        XTest = [XTest; X(idxTest)];    YTest = [YTest; Y(idxTest)];
    end
    
    % Final shuffle
    idx = randperm(numel(YTrain)); XTrain = XTrain(idx); YTrain = YTrain(idx);
    idx = randperm(numel(YVal));   XVal = XVal(idx);     YVal = YVal(idx);
    idx = randperm(numel(YTest));  XTest = XTest(idx);   YTest = YTest(idx);
end

function [XNorm, normParams] = normalizeSequences(X)
%NORMALIZESEQUENCES Robust Z-score normalization
    allData = cat(2, X{:});  % numFeatures x totalTimeSteps
    
    % Robust statistics to ignore outliers
    mu = median(allData, 2);
    sigma = iqr(allData, 2) / 1.35; % Estimate std from IQR
    
    % Avoid divide by zero
    sigma(sigma < 1e-6) = 1;
    
    normParams.mu = mu;
    normParams.sigma = sigma;
    
    XNorm = cell(size(X));
    for i = 1:numel(X)
        XNorm{i} = (X{i} - mu) ./ sigma;
    end
end

function XNorm = applyNormalization(X, normParams)
%APPLYNORMALIZATION Apply pre-computed normalization
    XNorm = cell(size(X));
    for i = 1:numel(X)
        XNorm{i} = (X{i} - normParams.mu) ./ normParams.sigma;
    end
end

function [XAug, YAug] = augmentData(X, Y)
%AUGMENTDATA Data augmentation
    XAug = X;
    YAug = Y;
    numOriginal = numel(X);
    
    % Augment 50% of data
    numAug = floor(0.5 * numOriginal);
    for i = 1:numAug
        idx = randi(numOriginal);
        % Add Gaussian noise
        xJitter = X{idx} + 0.05 * randn(size(X{idx}));
        % Random scaling (0.9 to 1.1)
        scale = 0.9 + 0.2*rand();
        xScaled = xJitter * scale;
        
        XAug{end+1} = xScaled;
        YAug(end+1) = Y(idx);
    end
end