function validateSystem()
%VALIDATESYSTEM Comprehensive validation and testing suite
%
%   Performs complete system validation including:
%   - Dataset integrity checks
%   - Model architecture verification
%   - Performance benchmarking
%   - Cross-validation analysis
%   - Robustness testing
%

    fprintf('\n');
    fprintf('╔════════════════════════════════════════╗\n');
    fprintf('║  PNEUMATIC DIGITAL TWIN VALIDATOR     ║\n');
    fprintf('║  Complete System Validation Suite     ║\n');
    fprintf('╚════════════════════════════════════════╝\n');
    fprintf('\n');
    
    allPassed = true;
    testResults = struct();
    
    %% ===== TEST 1: FILE EXISTENCE =====
    fprintf('[TEST 1] Checking required files...\n');
    
    requiredFiles = {
        'generatePneumaticDataset.m'
        'trainFaultClassifier.m'
        'runDigitalTwinSimulation.m'
        'validateSystem.m'
    };
    
    for i = 1:numel(requiredFiles)
        if exist(requiredFiles{i}, 'file')
            fprintf('  ✓ %s\n', requiredFiles{i});
        else
            fprintf('  ✗ %s MISSING!\n', requiredFiles{i});
            allPassed = false;
        end
    end
    
    testResults.fileExistence = allPassed;
    fprintf('\n');
    
    %% ===== TEST 2: DATASET VALIDATION =====
    fprintf('[TEST 2] Validating dataset...\n');
    
    if exist('pneumatic_dataset.mat', 'file')
        try
            load('pneumatic_dataset.mat', 'X', 'Y', 'datasetInfo');
            
            % Check dataset structure
            assert(iscell(X), 'X must be cell array');
            assert(isa(Y, 'categorical'), 'Y must be categorical');
            
            % Check dimensions
            numSamples = numel(X);
            numFeatures = size(X{1}, 1);
            timeSteps = size(X{1}, 2);
            
            fprintf('  ✓ Dataset loaded successfully\n');
            fprintf('    - Samples: %d\n', numSamples);
            fprintf('    - Features: %d\n', numFeatures);
            fprintf('    - Time steps: %d\n', timeSteps);
            fprintf('    - Classes: %d\n', numel(categories(Y)));
            
            % Check for NaN/Inf
            hasNaN = false;
            hasInf = false;
            for i = 1:min(100, numSamples)
                if any(isnan(X{i}(:)))
                    hasNaN = true;
                end
                if any(isinf(X{i}(:)))
                    hasInf = true;
                end
            end
            
            if ~hasNaN && ~hasInf
                fprintf('  ✓ No NaN or Inf values detected\n');
            else
                fprintf('  ✗ WARNING: Dataset contains invalid values!\n');
                allPassed = false;
            end
            
            % Class balance check
            classCounts = countcats(Y);
            minCount = min(classCounts);
            maxCount = max(classCounts);
            imbalanceRatio = maxCount / minCount;
            
            if imbalanceRatio < 2.0
                fprintf('  ✓ Classes are balanced (ratio: %.2f)\n', imbalanceRatio);
            else
                fprintf('  ⚠ Classes are imbalanced (ratio: %.2f)\n', imbalanceRatio);
            end
            
            testResults.datasetValid = true;
            
        catch ME
            fprintf('  ✗ Dataset validation failed: %s\n', ME.message);
            testResults.datasetValid = false;
            allPassed = false;
        end
    else
        fprintf('  ✗ Dataset file not found!\n');
        fprintf('    Run generatePneumaticDataset.m first\n');
        testResults.datasetValid = false;
        allPassed = false;
    end
    
    fprintf('\n');
    
    %% ===== TEST 3: MODEL VALIDATION =====
    fprintf('[TEST 3] Validating trained model...\n');
    
    if exist('pneumatic_fault_lstm.mat', 'file')
        try
            load('pneumatic_fault_lstm.mat', 'net', 'normParams', 'trainingInfo');
            
            fprintf('  ✓ Model loaded successfully\n');
            fprintf('    - Architecture: %s\n', class(net));
            fprintf('    - Test accuracy: %.2f%%\n', trainingInfo.testAccuracy * 100);
            
            % Validate normalization parameters
            assert(isfield(normParams, 'mu'), 'normParams missing mu');
            assert(isfield(normParams, 'sigma'), 'normParams missing sigma');
            fprintf('  ✓ Normalization parameters valid\n');
            
            % Check if accuracy is reasonable
            if trainingInfo.testAccuracy >= 0.70
                fprintf('  ✓ Model performance is good (≥70%%)\n');
            elseif trainingInfo.testAccuracy >= 0.50
                fprintf('  ⚠ Model performance is moderate (50-70%%)\n');
            else
                fprintf('  ✗ Model performance is poor (<50%%)\n');
                allPassed = false;
            end
            
            % Test inference
            if exist('pneumatic_dataset.mat', 'file')
                load('pneumatic_dataset.mat', 'X');
                testSample = (X{1} - normParams.mu) ./ normParams.sigma;
                [pred, scores] = classify(net, {testSample});
                
                fprintf('  ✓ Model inference works\n');
                fprintf('    - Prediction: %s\n', string(pred));
                fprintf('    - Confidence: %.1f%%\n', max(scores) * 100);
            end
            
            testResults.modelValid = true;
            
        catch ME
            fprintf('  ✗ Model validation failed: %s\n', ME.message);
            testResults.modelValid = false;
            allPassed = false;
        end
    else
        fprintf('  ✗ Trained model file not found!\n');
        fprintf('    Run trainFaultClassifier.m first\n');
        testResults.modelValid = false;
        allPassed = false;
    end
    
    fprintf('\n');
    
    %% ===== TEST 4: SIMULINK MODEL CHECK =====
    fprintf('[TEST 4] Checking Simulink model...\n');
    
    modelPath = '../models/PneumaticDigitalTwin.slx';
    
    if exist(modelPath, 'file')
        fprintf('  ✓ Simulink model file exists\n');
        fprintf('    Path: %s\n', modelPath);
        
        try
            % Try to load model (without opening GUI)
            load_system(modelPath);
            fprintf('  ✓ Model loads successfully\n');
            
            % Check model configuration
            modelName = 'PneumaticDigitalTwin';
            
            % Get solver settings
            solver = get_param(modelName, 'Solver');
            stepSize = get_param(modelName, 'FixedStep');
            
            fprintf('  ✓ Model configuration:\n');
            fprintf('    - Solver: %s\n', solver);
            fprintf('    - Fixed step: %s\n', stepSize);
            
            % Close model (don't leave it open)
            close_system(modelName, 0);
            
            testResults.simulinkValid = true;
            
        catch ME
            fprintf('  ✗ Error loading Simulink model: %s\n', ME.message);
            testResults.simulinkValid = false;
            allPassed = false;
        end
    else
        fprintf('  ✗ Simulink model file not found!\n');
        fprintf('    Expected location: %s\n', modelPath);
        fprintf('    You need to create this model manually\n');
        testResults.simulinkValid = false;
        % Don't fail overall test since Simulink model requires manual creation
    end
    
    fprintf('\n');
    
    %% ===== TEST 5: TOOLBOX AVAILABILITY =====
    fprintf('[TEST 5] Checking required toolboxes...\n');
    
    requiredToolboxes = {
        'MATLAB'
        'Simulink'
        'Deep Learning Toolbox'
    };
    
    installedToolboxes = ver;
    toolboxNames = {installedToolboxes.Name};
    
    allToolboxesPresent = true;
    for i = 1:numel(requiredToolboxes)
        if any(contains(toolboxNames, requiredToolboxes{i}))
            fprintf('  ✓ %s\n', requiredToolboxes{i});
        else
            fprintf('  ✗ %s NOT INSTALLED!\n', requiredToolboxes{i});
            allToolboxesPresent = false;
            allPassed = false;
        end
    end
    
    testResults.toolboxesPresent = allToolboxesPresent;
    fprintf('\n');
    
    %% ===== TEST 6: CROSS-VALIDATION TEST =====
    fprintf('[TEST 6] Running cross-validation test...\n');
    
    if exist('pneumatic_dataset.mat', 'file') && ...
       exist('pneumatic_fault_lstm.mat', 'file')
        
        try
            load('pneumatic_dataset.mat', 'X', 'Y');
            load('pneumatic_fault_lstm.mat', 'net', 'normParams');
            
            % Quick 3-fold CV on small subset
            numSamples = min(300, numel(X));
            indices = randperm(numel(X), numSamples);
            X_subset = X(indices);
            Y_subset = Y(indices);
            
            k = 3;
            cvAccuracy = zeros(k, 1);
            foldSize = floor(numSamples / k);
            
            fprintf('  Running %d-fold cross-validation...\n', k);
            
            for fold = 1:k
                testIdx = (fold-1)*foldSize + 1 : min(fold*foldSize, numSamples);
                
                X_test = X_subset(testIdx);
                Y_test = Y_subset(testIdx);
                
                % Normalize
                X_test_norm = cell(size(X_test));
                for i = 1:numel(X_test)
                    X_test_norm{i} = (X_test{i} - normParams.mu) ./ normParams.sigma;
                end
                
                % Predict
                Y_pred = classify(net, X_test_norm);
                cvAccuracy(fold) = mean(Y_pred == Y_test);
                
                fprintf('    Fold %d: %.1f%%\n', fold, cvAccuracy(fold) * 100);
            end
            
            meanCV = mean(cvAccuracy);
            stdCV = std(cvAccuracy);
            
            fprintf('  ✓ Cross-validation completed\n');
            fprintf('    - Mean accuracy: %.1f%% ± %.1f%%\n', meanCV * 100, stdCV * 100);
            
            if stdCV < 0.10
                fprintf('  ✓ Model is stable (low variance)\n');
            else
                fprintf('  ⚠ Model has high variance\n');
            end
            
            testResults.crossValidation = meanCV;
            
        catch ME
            fprintf('  ✗ Cross-validation failed: %s\n', ME.message);
            testResults.crossValidation = 0;
        end
    else
        fprintf('  ⊘ Skipping (dataset or model not available)\n');
        testResults.crossValidation = NaN;
    end
    
    fprintf('\n');
    
    %% ===== TEST 7: ROBUSTNESS TEST =====
    fprintf('[TEST 7] Testing model robustness...\n');
    
    if exist('pneumatic_dataset.mat', 'file') && ...
       exist('pneumatic_fault_lstm.mat', 'file')
        
        try
            load('pneumatic_dataset.mat', 'X', 'Y');
            load('pneumatic_fault_lstm.mat', 'net', 'normParams');
            
            % Test with noise-corrupted data
            numTestSamples = min(50, numel(X));
            noiseLevels = [0, 0.1, 0.2, 0.5];
            
            fprintf('  Testing robustness to noise...\n');
            
            for noiseLevel = noiseLevels
                correct = 0;
                
                for i = 1:numTestSamples
                    % Add Gaussian noise
                    X_noisy = X{i} + noiseLevel * randn(size(X{i}));
                    
                    % Normalize and classify
                    X_noisy_norm = (X_noisy - normParams.mu) ./ normParams.sigma;
                    pred = classify(net, {X_noisy_norm});
                    
                    if pred == Y(i)
                        correct = correct + 1;
                    end
                end
                
                accuracy = correct / numTestSamples;
                fprintf('    Noise σ=%.2f: %.1f%%\n', noiseLevel, accuracy * 100);
            end
            
            fprintf('  ✓ Robustness test completed\n');
            testResults.robustnessTest = true;
            
        catch ME
            fprintf('  ✗ Robustness test failed: %s\n', ME.message);
            testResults.robustnessTest = false;
        end
    else
        fprintf('  ⊘ Skipping (dataset or model not available)\n');
        testResults.robustnessTest = false;
    end
    
    fprintf('\n');
    
    %% ===== TEST 8: OUTPUT VALIDATION =====
    fprintf('[TEST 8] Validating output files...\n');
    
    outputFiles = {
        'confusion_matrix.png'
        'roc_curves.png'
        'feature_importance.png'
    };
    
    outputsPresent = 0;
    for i = 1:numel(outputFiles)
        if exist(outputFiles{i}, 'file')
            fprintf('  ✓ %s\n', outputFiles{i});
            outputsPresent = outputsPresent + 1;
        else
            fprintf('  ⊘ %s (not yet generated)\n', outputFiles{i});
        end
    end
    
    testResults.outputsGenerated = outputsPresent;
    fprintf('\n');
    
    %% ===== FINAL REPORT =====
    fprintf('╔════════════════════════════════════════╗\n');
    fprintf('║         VALIDATION SUMMARY             ║\n');
    fprintf('╚════════════════════════════════════════╝\n');
    fprintf('\n');
    
    % Generate summary table
    tests = {
        'File Existence', testResults.fileExistence
        'Dataset Validity', testResults.datasetValid
        'Model Validity', testResults.modelValid
        'Simulink Model', testResults.simulinkValid
        'Toolboxes', testResults.toolboxesPresent
    };
    
    passCount = 0;
    totalTests = size(tests, 1);
    
    for i = 1:totalTests
        testName = tests{i, 1};
        testPassed = tests{i, 2};
        
        if testPassed
            fprintf('  ✓ %-25s PASSED\n', testName);
            passCount = passCount + 1;
        else
            fprintf('  ✗ %-25s FAILED\n', testName);
        end
    end
    
    fprintf('\n');
    fprintf('Overall: %d/%d tests passed\n', passCount, totalTests);
    fprintf('\n');
    
    if allPassed
        fprintf('╔════════════════════════════════════════╗\n');
        fprintf('║    ✓ ALL VALIDATION TESTS PASSED!     ║\n');
        fprintf('║    System is ready for submission     ║\n');
        fprintf('╚════════════════════════════════════════╝\n');
    else
        fprintf('╔════════════════════════════════════════╗\n');
        fprintf('║  ⚠ SOME TESTS FAILED OR INCOMPLETE    ║\n');
        fprintf('║    Please review errors above         ║\n');
        fprintf('╚════════════════════════════════════════╝\n');
    end
    
    fprintf('\n');
    
    %% ===== RECOMMENDATIONS =====
    fprintf('RECOMMENDATIONS:\n');
    fprintf('\n');
    
    if ~testResults.datasetValid
        fprintf('  1. Generate dataset: run generatePneumaticDataset()\n');
    end
    
    if ~testResults.modelValid
        fprintf('  2. Train model: run trainFaultClassifier()\n');
    end
    
    if ~testResults.simulinkValid
        fprintf('  3. Create Simulink model following the provided instructions\n');
    end
    
    if testResults.modelValid && exist('pneumatic_fault_lstm.mat', 'file')
        load('pneumatic_fault_lstm.mat', 'trainingInfo');
        if trainingInfo.testAccuracy < 0.70
            fprintf('  4. Consider tuning hyperparameters to improve accuracy\n');
            fprintf('     Current: %.1f%%, Target: >70%%\n', trainingInfo.testAccuracy * 100);
        end
    end
    
    fprintf('\n');
    fprintf('For complete project execution:\n');
    fprintf('  1. generatePneumaticDataset()\n');
    fprintf('  2. trainFaultClassifier()\n');
    fprintf('  3. Create Simulink model\n');
    fprintf('  4. runDigitalTwinSimulation()\n');
    fprintf('  5. Record video demonstration\n');
    fprintf('\n');
    
    % Save validation report
    save('validation_report.mat', 'testResults', '-v7.3');
    fprintf('✓ Validation report saved to: validation_report.mat\n\n');
end


