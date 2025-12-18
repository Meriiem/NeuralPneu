function runDigitalTwinSimulation()
%RUNDIGITALTWINSIMULATION Complete digital twin simulation with ML classification
%
%   Features:
%   - Automated Simulink model execution
%   - Real-time fault classification
%   - Comprehensive visualization dashboard
%   - Performance analysis and reporting
%   - Export results for video presentation
%
%   Prerequisites:
%       - PneumaticDigitalTwin.slx in models/
%       - pneumatic_fault_lstm.mat (trained model)
%       - MATLAB R2020b or later
%       - Simulink, Deep Learning Toolbox

    fprintf('\n========================================\n');
    fprintf('PNEUMATIC DIGITAL TWIN SIMULATOR\n');
    fprintf('========================================\n\n');
    
    %% ===== INITIALIZATION =====
    fprintf('Initializing system...\n');
    
    % Check prerequisites
    checkPrerequisites();
    
    modelFile = fullfile('models', 'PneumaticDigitalTwin.slx');
    modelName = 'PneumaticDigitalTwin';
    
    if ~bdIsLoaded(modelName)
        load_system(modelFile);
    end

    % Load trained model
    fprintf('  Loading trained classifier...\n');
    load('pneumatic_fault_lstm.mat', 'net', 'normParams', 'trainingInfo', 'datasetInfo');
    
    fprintf('  ✓ Model loaded (Test accuracy: %.2f%%)\n', ...
        trainingInfo.testAccuracy * 100);
    
    % Make available to Simulink
    assignin('base', 'net', net);
    assignin('base', 'normParams', normParams);
    assignin('base', 'Ts', datasetInfo.config.Ts);
    

    fprintf('\n');
    
    %% ===== FAULT SCENARIOS =====
    faultScenarios = [
        struct('id', uint8(0), 'name', "normal", 'description', "Normal operation - no faults")
        struct('id', uint8(1), 'name', "leak_small", 'description', "Small leak in pneumatic line")
        struct('id', uint8(2), 'name', "leak_large", 'description', "Large leak - significant pressure loss")
        struct('id', uint8(3), 'name', "leak_critical", 'description', "Critical leak - severe pressure drop")
        struct('id', uint8(4), 'name', "valve_stuck_half", 'description', "Control valve stuck at 50% open")
        struct('id', uint8(5), 'name', "valve_stuck_closed", 'description', "Control valve stuck closed")
        struct('id', uint8(6), 'name', "sensor_bias_pos", 'description', "Pressure sensor positive bias")
        struct('id', uint8(7), 'name', "sensor_bias_neg", 'description', "Pressure sensor negative bias")
    ];
    
    numScenarios = numel(faultScenarios);
    
    %% ===== SIMULATION CONFIGURATION =====
    simConfig.stopTime = '20';      % 20 seconds
    simConfig.solver = 'ode4';       % Fixed-step 4th order Runge-Kutta
    simConfig.fixedStep = '0.02';    % 20 ms
    
    fprintf('========================================\n');
    fprintf('RUNNING SIMULATIONS\n');
    fprintf('========================================\n\n');
    fprintf('Simulation config:\n');
    fprintf('  Duration: %s seconds\n', simConfig.stopTime);
    fprintf('  Solver: %s\n', simConfig.solver);
    fprintf('  Step size: %s seconds\n', simConfig.fixedStep);
    fprintf('  Scenarios: %d\n', numScenarios);
    fprintf('\n');
    
    %% ===== RUN SIMULATIONS =====
    results = cell(numScenarios, 1);
    
    for i = 1:numScenarios
        scenario = faultScenarios(i);
        
        fprintf('[%d/%d] Simulating: %s\n', i, numScenarios, scenario.name);
        fprintf('       %s\n', scenario.description);
        
        % Set fault ID in base workspace
        assignin('base', 'faultId', scenario.id);
        
        % Configure model
        set_param(modelName, 'StopTime', simConfig.stopTime);
        set_param(modelName, 'Solver', simConfig.solver);
        set_param(modelName, 'FixedStep', num2str(simConfig.fixedStep));        
        
        % Run simulation
        tic;
        simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');
        simTime = toc;
        
        fprintf('       Simulation completed in %.2f seconds\n', simTime);
        
        % Extract logged signals
        result = extractSimulationData(simOut, scenario, datasetInfo.config.WINDOW);
        result.simTime = simTime;
        
        % Classify fault using trained network
        result = classifyFault(result, net, normParams, trainingInfo.classNames);
        
        fprintf('       Predicted: %s (Confidence: %.1f%%)\n', ...
            result.predictedFault, result.confidence * 100);
        fprintf('\n');
        
        results{i} = result;
    end
    
    %% ===== PERFORMANCE ANALYSIS =====
    fprintf('========================================\n');
    fprintf('PERFORMANCE ANALYSIS\n');
    fprintf('========================================\n\n');
    
    analyzePerformance(results, faultScenarios);
    
    %% ===== VISUALIZATION =====
    fprintf('\n========================================\n');
    fprintf('GENERATING VISUALIZATIONS\n');
    fprintf('========================================\n\n');
    
    visualizeResults(results, faultScenarios);
    
    fprintf('✓ Visualizations saved\n\n');
    
    %% ===== EXPORT RESULTS =====
    fprintf('========================================\n');
    fprintf('EXPORTING RESULTS\n');
    fprintf('========================================\n\n');
    
    exportResults(results, faultScenarios, trainingInfo);
    
    fprintf('✓ Results exported\n\n');
    
    %% ===== SUMMARY =====
    fprintf('========================================\n');
    fprintf('SIMULATION COMPLETE\n');
    fprintf('========================================\n\n');
    
    resultsStruct = [results{:}];

    fprintf('Summary:\n');
    fprintf('  Total scenarios simulated: %d\n', numScenarios);
    fprintf('  Correct classifications: %d/%d (%.1f%%)\n', ...
        sum([resultsStruct.correctClassification]), numScenarios, ...
        100 * mean([resultsStruct.correctClassification]));
    fprintf('  Average confidence: %.1f%%\n', ...
        100 * mean([resultsStruct.confidence]));
    fprintf('\n');

    
    fprintf('Files generated:\n');
    fprintf('  - simulation_results.mat\n');
    fprintf('  - simulation_dashboard.png\n');
    fprintf('  - individual_scenarios/*.png\n');
    fprintf('  - performance_report.pdf (if available)\n');
    fprintf('\n');
    
    fprintf('Next steps:\n');
    fprintf('  1. Review visualizations\n');
    fprintf('  2. Record video demonstration\n');
    fprintf('  3. Prepare presentation slides\n');
    fprintf('\n');
    
    fprintf('========================================\n\n');
end

%% ===== HELPER FUNCTIONS =====

function checkPrerequisites()
%CHECKPREREQUISITES Verify all required files and toolboxes

    % Check model file
    modelPath = 'models/PneumaticDigitalTwin.slx';
    if ~exist(modelPath, 'file')
        error(['Simulink model not found: %s\n' ...
               'Please create the model first using the provided instructions.'], modelPath);
    end
    
    % Check trained network
    if ~exist('pneumatic_fault_lstm.mat', 'file')
        error(['Trained model not found: pneumatic_fault_lstm.mat\n' ...
               'Please run trainFaultClassifier.m first.']);
    end
    
    % Check toolboxes
    requiredToolboxes = {'Simulink', 'Deep Learning Toolbox'};
    installedToolboxes = ver;
    
    for i = 1:numel(requiredToolboxes)
        if ~any(strcmp({installedToolboxes.Name}, requiredToolboxes{i}))
            error('Required toolbox not installed: %s', requiredToolboxes{i});
        end
    end
    
    fprintf('  ✓ All prerequisites satisfied\n');
end

function result = extractSimulationData(simOut, scenario, windowSize)
%EXTRACTSIMULATIONDATA Extract and process simulation outputs

    % Get logged data
    if isprop(simOut, 'logsout') && ~isempty(simOut.logsout)
        logs = simOut.logsout;
        
        % Extract time series
        P_meas = logs.get('P_meas');
        P_twin = logs.get('P_twin');
        residual = logs.get('residual');
        u_valve = logs.get('u_valve');
        
        % Convert to arrays
        result.time = P_meas.Values.Time;
        result.P_meas = P_meas.Values.Data;
        result.P_twin = P_twin.Values.Data;
        result.residual = residual.Values.Data;
        result.u_valve = u_valve.Values.Data;
    else
        % Fallback: get from workspace variables
        result.time = evalin('base', 'tout');
        result.P_meas = evalin('base', 'P_meas_log');
        result.P_twin = evalin('base', 'P_twin_log');
        result.u_valve = evalin('base', 'u_log');
        result.residual = result.P_meas - result.P_twin;
    end
    
    % Compute additional features
    Ts = mean(diff(result.time));
    result.dP_meas = [0; diff(result.P_meas)] / Ts;
    result.dP_twin = [0; diff(result.P_twin)] / Ts;
    result.residual_MA = movmean(result.residual, 10);
    
    % Extract feature window for classification
    N = numel(result.time);
    if N >= windowSize
        faultStart = find(abs(result.residual) > 0.05*max(abs(result.residual)), 1);

        if isempty(faultStart)
            idxStart = max(1, N - windowSize + 1);
        else
            idxStart = max(faultStart, N - windowSize + 1);
        end

        result.featureWindow = [
            result.P_meas(idxStart:end)';
            result.P_twin(idxStart:end)';
            result.residual(idxStart:end)';
            result.u_valve(idxStart:end)';
            result.dP_meas(idxStart:end)';
            result.dP_twin(idxStart:end)';
            result.residual_MA(idxStart:end)'
        ];
    else
        warning('Simulation too short for feature window, using all data');
        result.featureWindow = [
            result.P_meas';
            result.P_twin';
            result.residual';
            result.u_valve';
            result.dP_meas';
            result.dP_twin';
            result.residual_MA'
        ];
    end
    
    % Store scenario info
    result.trueFault = scenario.name;
    result.faultId = scenario.id;
    result.description = scenario.description;
end

function result = classifyFault(result, net, normParams, classNames)
%CLASSIFYFAULT Classify fault from feature window

    % Normalize features
    featureNorm = (result.featureWindow - normParams.mu) ./ normParams.sigma;
    
    % Classify
    [predictedLabel, scores] = classify(net, {featureNorm}, 'MiniBatchSize', 1);
    
    result.predictedFault = string(predictedLabel);
    result.classScores = scores;
    result.confidence = max(scores);
    
    % Check if correct
    result.correctClassification = (result.predictedFault == result.trueFault);
    
    % Store all class probabilities
    result.classProbabilities = struct();
    for i = 1:numel(classNames)
        result.classProbabilities.(matlab.lang.makeValidName(string(classNames{i}))) = scores(i);
    end
end

function analyzePerformance(results, scenarios)
%ANALYZEPERFORMANCE Detailed performance analysis

    resultsStruct = [results{:}];
    numScenarios = numel(scenarios);
    
    fprintf('Classification Results:\n');
    fprintf('%-20s %-20s %12s %12s\n', 'True Fault', 'Predicted', 'Confidence', 'Status');
    fprintf('%s\n', repmat('-', 1, 70));
    
    for i = 1:numScenarios
        r = results{i};
        status = '✓ CORRECT';
        if ~r.correctClassification
            status = '✗ WRONG';
        end
        
        fprintf('%-20s %-20s %11.1f%% %12s\n', ...
            r.trueFault, r.predictedFault, r.confidence * 100, status);
    end
    
    fprintf('%s\n\n', repmat('-', 1, 70));
        
    % Compute metrics
    accuracy = mean([resultsStruct.correctClassification]);
    avgConfidence = mean([resultsStruct.confidence]);
    
    fprintf('Metrics:\n');
    fprintf('  Overall accuracy: %.1f%%\n', accuracy * 100);
    fprintf('  Average confidence: %.1f%%\n', avgConfidence * 100);
    fprintf('  Correct predictions: %d/%d\n', ...
        sum([resultsStruct.correctClassification]), numScenarios);
    end


function visualizeResults(results, scenarios)

    % ===== VERY SIMPLE + FAST VISUALIZATION =====
    % One figure, first scenario only (for recording/demo)

    r = results{1};   % show ONLY first scenario (normal / example)

    % Downsample hard
    MAX_PLOT_POINTS = 200;
    idx = round(linspace(1, numel(r.time), ...
        min(MAX_PLOT_POINTS, numel(r.time))));

    fig = figure('Visible','off','Position',[100 100 1000 600],'Color','w');

    plot(r.time(idx), r.P_meas(idx)/1e5, 'LineWidth',2); hold on;
    plot(r.time(idx), r.P_twin(idx)/1e5, '--', 'LineWidth',2);

    xlabel('Time (s)');
    ylabel('Pressure (bar)');
    title(sprintf('Digital Twin | True: %s | Pred: %s (%.1f%%)', ...
        r.trueFault, r.predictedFault, r.confidence*100), ...
        'Interpreter','none');

    legend('Measured','Digital Twin','Location','best');
    grid on;

    exportgraphics(fig, 'simulation_dashboard.png', 'Resolution', 120);
    close(fig);
end


function exportResults(results, scenarios, trainingInfo)
%EXPORTRESULTS Export results to MAT file and generate report

    % Save all results
    save('simulation_results.mat', 'results', 'scenarios', 'trainingInfo', '-v7.3');
    
    % Generate text report
    fid = fopen('simulation_report.txt', 'w');
    
    fprintf(fid, '========================================\n');
    fprintf(fid, 'PNEUMATIC DIGITAL TWIN SIMULATION REPORT\n');
    fprintf(fid, '========================================\n\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(now));
    
    fprintf(fid, 'Model Information:\n');
    fprintf(fid, '  Test Accuracy: %.2f%%\n', trainingInfo.testAccuracy * 100);
    fprintf(fid, '  Training Samples: %d\n', trainingInfo.numTrainSamples);
    fprintf(fid, '  Test Samples: %d\n\n', trainingInfo.numTestSamples);
    
    fprintf(fid, 'Simulation Results:\n');
    fprintf(fid, '%-20s %-20s %12s %12s\n', 'True Fault', 'Predicted', 'Confidence', 'Status');
    fprintf(fid, '%s\n', repmat('-', 1, 70));
        
    for i = 1:numel(results)
        r = results{i};
    
        if r.correctClassification
            status = 'CORRECT';
        else
            status = 'WRONG';
        end
    
        fprintf(fid, '%-20s %-20s %11.1f%% %12s\n', ...
            char(r.trueFault), char(r.predictedFault), r.confidence * 100, status);
    end
    
    fprintf(fid, '\n========================================\n');
    
    fclose(fid);
end