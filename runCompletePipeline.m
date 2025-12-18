function runCompletePipeline(varargin)
%RUNCOMPLETEPIPELINE Automated execution of entire project pipeline
%
%   This script automates the complete workflow:
%   1. Dataset generation
%   2. Model training
%   3. System validation
%   4. (Optional) Simulink simulation
%
%   Perfect for first-time setup or regenerating all results
%
%   Usage:
%       runCompletePipeline()           % Run everything
%       runCompletePipeline('quick')    % Skip Simulink simulation

    clc;
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                                                          ║\n');
    fprintf('║     PNEUMATIC DIGITAL TWIN - COMPLETE PIPELINE          ║\n');
    fprintf('║     Simulink Student Challenge 2025                     ║\n');
    fprintf('║                                                          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    
    %% ===== CONFIGURATION =====
    config = struct();
    config.skipSimulink = false;
    config.generateDataset = true;
    config.trainModel = true;
    config.runValidation = true;
    
    % Parse arguments
    if nargin > 0 && strcmpi(varargin{1}, 'quick')
        config.skipSimulink = true;
        fprintf('⚡ Quick mode: Skipping Simulink simulation\n\n');
    end
    
    %% ===== PRE-FLIGHT CHECKS =====
    fprintf('═══════════════════════════════════════\n');
    fprintf('STEP 0: Pre-flight Checks\n');
    fprintf('═══════════════════════════════════════\n\n');
    
    % Check MATLAB version
    matlabVer = version('-release');
    fprintf('MATLAB version: %s\n', matlabVer);
    
    if str2double(matlabVer(1:4)) < 2020
        warning('MATLAB R2020b or later is recommended');
    else
        fprintf('✓ MATLAB version compatible\n');
    end
    
    % Check toolboxes
    fprintf('\nChecking required toolboxes...\n');
    requiredToolboxes = {'Deep Learning Toolbox', 'Simulink'};
    allPresent = true;
    
    installedToolboxes = ver;
    for i = 1:numel(requiredToolboxes)
        if any(contains({installedToolboxes.Name}, requiredToolboxes{i}))
            fprintf('  ✓ %s\n', requiredToolboxes{i});
        else
            fprintf('  ✗ %s NOT FOUND\n', requiredToolboxes{i});
            allPresent = false;
        end
    end
    
    if ~allPresent
        error('Missing required toolboxes. Please install them first.');
    end
    
    fprintf('\n✓ All prerequisites satisfied\n');
    
    % Confirmation
    fprintf('\n');
    fprintf('This pipeline will:\n');
    fprintf('  1. Generate dataset (~5-10 min)\n');
    fprintf('  2. Train classifier (~10-20 min)\n');
    fprintf('  3. Run validation tests\n');
    if ~config.skipSimulink
        fprintf('  4. Execute Simulink simulations (requires manual model setup)\n');
    end

    if config.skipSimulink
        estTime = '15-30 minutes';
    else
        estTime = '20-35 minutes';
    end
    
    fprintf('\nEstimated total time: %s\n', estTime);

    fprintf('\n');
    
    response = input('Continue? (Y/n): ', 's');
    if ~isempty(response) && ~strcmpi(response, 'y')
        fprintf('\nPipeline cancelled.\n');
        return;
    end
    
    fprintf('\n');
    
    %% ===== PIPELINE EXECUTION =====
    pipelineTimer = tic;
    results = struct();
    
    %% Step 1: Dataset Generation
    if config.generateDataset
        fprintf('\n');
        fprintf('═══════════════════════════════════════\n');
        fprintf('STEP 1: Dataset Generation\n');
        fprintf('═══════════════════════════════════════\n\n');
        
        try
            stepTimer = tic;
            generatePneumaticDataset();
            results.datasetTime = toc(stepTimer);
            results.datasetSuccess = true;
            
            fprintf('\n✓ Dataset generation completed in %.2f seconds\n', results.datasetTime);
            
        catch ME
            fprintf('\n✗ Dataset generation failed!\n');
            fprintf('Error: %s\n', ME.message);
            results.datasetSuccess = false;
            rethrow(ME);
        end
    end
    
    %% Step 2: Model Training
    if config.trainModel
        fprintf('\n');
        fprintf('═══════════════════════════════════════\n');
        fprintf('STEP 2: Classifier Training\n');
        fprintf('═══════════════════════════════════════\n\n');
        
        if ~exist('pneumatic_dataset.mat', 'file')
            error('Dataset not found! Cannot train model.');
        end
        
        try
            stepTimer = tic;
            trainFaultClassifier();
            results.trainingTime = toc(stepTimer);
            results.trainingSuccess = true;
            
            % Load and display accuracy
            load('pneumatic_fault_lstm.mat', 'trainingInfo');
            results.testAccuracy = trainingInfo.testAccuracy;
            
            fprintf('\n✓ Model training completed in %.2f seconds\n', results.trainingTime);
            fprintf('✓ Test accuracy: %.2f%%\n', results.testAccuracy * 100);
            
            if results.testAccuracy < 0.70
                fprintf('\n⚠ Warning: Accuracy is below 70%%\n');
                fprintf('  Consider retraining with different parameters\n');
            end
            
        catch ME
            fprintf('\n✗ Model training failed!\n');
            fprintf('Error: %s\n', ME.message);
            results.trainingSuccess = false;
            rethrow(ME);
        end
    end
    
    %% Step 3: Validation
    if config.runValidation
        fprintf('\n');
        fprintf('═══════════════════════════════════════\n');
        fprintf('STEP 3: System Validation\n');
        fprintf('═══════════════════════════════════════\n\n');
        
        try
            validateSystem();
            results.validationSuccess = true;
            
        catch ME
            fprintf('\n⚠ Validation completed with warnings\n');
            fprintf('Error: %s\n', ME.message);
            results.validationSuccess = false;
        end
    end
    
    %% Step 4: Simulink Simulation (Optional)
    if ~config.skipSimulink
        fprintf('\n');
        fprintf('═══════════════════════════════════════\n');
        fprintf('STEP 4: Simulink Simulation\n');
        fprintf('═══════════════════════════════════════\n\n');
        
        modelPath = '../models/PneumaticDigitalTwin.slx';
        
        if ~exist(modelPath, 'file')
            fprintf('⊘ Simulink model not found at: %s\n', modelPath);
            fprintf('  You need to create the Simulink model manually\n');
            fprintf('  See README.md for detailed instructions\n');
            results.simulinkSuccess = false;
        else
            try
                stepTimer = tic;
                runDigitalTwinSimulation();
                results.simulationTime = toc(stepTimer);
                results.simulinkSuccess = true;
                
                fprintf('\n✓ Simulink simulation completed in %.2f seconds\n', ...
                    results.simulationTime);
                
            catch ME
                fprintf('\n✗ Simulink simulation failed!\n');
                fprintf('Error: %s\n', ME.message);
                results.simulinkSuccess = false;
            end
        end
    end
    
    %% ===== PIPELINE SUMMARY =====
    totalTime = toc(pipelineTimer);
    
    fprintf('\n\n');
    fprintf('╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                  PIPELINE COMPLETED                      ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    
    fprintf('Execution Summary:\n');
    fprintf('─────────────────────────────────────────────────────────\n');
    
    if isfield(results, 'datasetSuccess') && results.datasetSuccess
        fprintf('  ✓ Dataset Generation:    %.1f seconds\n', results.datasetTime);
    end
    
    if isfield(results, 'trainingSuccess') && results.trainingSuccess
        fprintf('  ✓ Model Training:        %.1f seconds (Accuracy: %.1f%%)\n', ...
            results.trainingTime, results.testAccuracy * 100);
    end
    
    if isfield(results, 'validationSuccess')
        if results.validationSuccess
            fprintf('  ✓ System Validation:     PASSED\n');
        else
            fprintf('  ⚠ System Validation:     WARNINGS\n');
        end
    end
    
    if isfield(results, 'simulinkSuccess')
        if results.simulinkSuccess
            fprintf('  ✓ Simulink Simulation:   %.1f seconds\n', results.simulationTime);
        else
            fprintf('  ⊘ Simulink Simulation:   SKIPPED/FAILED\n');
        end
    end
    
    fprintf('─────────────────────────────────────────────────────────\n');
    fprintf('Total execution time: %.1f seconds (%.2f minutes)\n', ...
        totalTime, totalTime/60);
    fprintf('\n');
    
    %% ===== FILES GENERATED =====
    fprintf('Files Generated:\n');
    fprintf('─────────────────────────────────────────────────────────\n');
    
    generatedFiles = {
        'pneumatic_dataset.mat', 'Dataset'
        'pneumatic_fault_lstm.mat', 'Trained model'
        'confusion_matrix.png', 'Confusion matrix'
        'roc_curves.png', 'ROC curves'
        'feature_importance.png', 'Feature importance'
        'validation_report.mat', 'Validation results'
        'simulation_results.mat', 'Simulation outputs'
        'simulation_dashboard.png', 'Visualization dashboard'
    };
    
    for i = 1:size(generatedFiles, 1)
        filename = generatedFiles{i, 1};
        description = generatedFiles{i, 2};
        
        if exist(filename, 'file')
            info = dir(filename);
            fprintf('  ✓ %-30s (%.2f MB) - %s\n', ...
                filename, info.bytes/1e6, description);
        else
            fprintf('  ⊘ %-30s - %s (not generated)\n', ...
                filename, description);
        end
    end
    
    fprintf('\n');
    
    %% ===== NEXT STEPS =====
    fprintf('╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                      NEXT STEPS                          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    
    nextSteps = {};
    stepNum = 1;
    
    if ~exist('../models/PneumaticDigitalTwin.slx', 'file')
        nextSteps{end+1} = sprintf('%d. Create Simulink model (see README.md)', stepNum);
        stepNum = stepNum + 1;
    end
    
    if isfield(results, 'testAccuracy') && results.testAccuracy < 0.75
        nextSteps{end+1} = sprintf('%d. Improve model accuracy (currently %.1f%%)', ...
            stepNum, results.testAccuracy * 100);
        stepNum = stepNum + 1;
    end
    
    nextSteps{end+1} = sprintf('%d. Review all generated visualizations', stepNum);
    stepNum = stepNum + 1;
    
    nextSteps{end+1} = sprintf('%d. Record video demonstration (3-5 minutes)', stepNum);
    stepNum = stepNum + 1;
    
    nextSteps{end+1} = sprintf('%d. Upload to YouTube with #SimulinkStudentChallenge2025', stepNum);
    stepNum = stepNum + 1;
    
    nextSteps{end+1} = sprintf('%d. Submit entry form before deadline', stepNum);
    
    for i = 1:numel(nextSteps)
        fprintf('  %s\n', nextSteps{i});
    end
    
    fprintf('\n');
    
    %% ===== RECOMMENDATIONS =====
    fprintf('Recommendations:\n');
    fprintf('─────────────────────────────────────────────────────────\n');
    
    if isfield(results, 'testAccuracy')
        if results.testAccuracy >= 0.80
            fprintf('  ✓ Excellent accuracy! Ready for video demonstration\n');
        elseif results.testAccuracy >= 0.70
            fprintf('  ✓ Good accuracy. Consider minor tuning for better results\n');
        else
            fprintf('  ⚠ Accuracy could be improved. Try:\n');
            fprintf('     - Increase training epochs to 80-100\n');
            fprintf('     - Add more samples: nPerFault = 300\n');
            fprintf('     - Reduce sensor noise in dataset generator\n');
        end
    end
    
    fprintf('\n');
    fprintf('For video recording:\n');
    fprintf('  • Show this script output and visualizations\n');
    fprintf('  • Demonstrate Simulink model with different faults\n');
    fprintf('  • Explain the digital twin concept clearly\n');
    fprintf('  • Keep it under 5 minutes, be enthusiastic!\n');
    fprintf('\n');
    
    %% ===== SAVE PIPELINE RESULTS =====
    results.totalTime = totalTime;
    results.timestamp = datetime('now');
    save('pipeline_results.mat', 'results', '-v7.3');
    
    fprintf('✓ Pipeline results saved to: pipeline_results.mat\n');
    fprintf('\n');
    
    fprintf('╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║                 🎉 ALL DONE! GOOD LUCK! 🎉               ║\n');
    fprintf('╚══════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
end