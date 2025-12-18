function generatePneumaticDataset()
%GENERATEPNEUMATICDATASET Advanced pneumatic system dataset generator
%   Generates high-quality, diverse training data for fault classification
%   with multiple operational scenarios, realistic physics, and noise models
%
%   MODIFICATIONS FOR STABILITY:
%   - Implemented Physics Sub-stepping (Oversampling) to prevent RK4 explosion
%   - Added Pre-derivative Smoothing to clean up dP/dt features
%   - Clamped physics values to realistic physical ranges
%
%   Output:
%       pneumatic_dataset.mat containing:
%           X  - cell array of sequences (numSamples x 1)
%           Y  - categorical labels (numSamples x 1)
%           metadata - simulation details
    fprintf('\n========================================\n');
    fprintf('PNEUMATIC DIGITAL TWIN DATASET GENERATOR\n');
    fprintf('========================================\n\n');
    
    rng(42);  % Reproducibility
    
    %% ===== SIMULATION CONFIGURATION =====
    config.Ts   = 0.02;      % 20 ms sample time (50 Hz) - Sensor Rate
    config.Tsim = 20;        % 20 second simulations
    config.t    = 0:config.Ts:config.Tsim;
    config.N    = numel(config.t);
    
    % Feature extraction window (last N samples for steady-state)
    config.WINDOW = 400;     % 8 seconds at 50 Hz
    
    if config.WINDOW > config.N
        error('WINDOW size exceeds simulation length!');
    end
    
    %% ===== FAULT TAXONOMY =====
    % 7 distinct fault classes with industrial relevance
    faultTypes = [
        "normal",           % Baseline operation
        "leak_small",       % Gradual pressure loss
        "leak_large",       % Rapid pressure loss  
        "leak_critical",    % Severe leak (new)
        "valve_stuck_half", % Stuck at 50% open
        "valve_stuck_closed",% Stuck closed (new)
        "sensor_bias_pos",  % Positive bias
        "sensor_bias_neg"   % Negative bias (new)
    ];
    
    nPerFault  = 200;  % Samples per class
    numSamples = numel(faultTypes) * nPerFault;
    
    fprintf('Configuration:\n');
    fprintf('  Sample time: %.3f s (%.1f Hz)\n', config.Ts, 1/config.Ts);
    fprintf('  Simulation time: %.1f s\n', config.Tsim);
    fprintf('  Feature window: %d samples (%.1f s)\n', config.WINDOW, config.WINDOW*config.Ts);
    fprintf('  Fault classes: %d\n', numel(faultTypes));
    fprintf('  Samples per class: %d\n', nPerFault);
    fprintf('  Total samples: %d\n', numSamples);
    fprintf('\n');
    
    %% ===== DATA STRUCTURES =====
    X = cell(numSamples, 1);           % Feature sequences
    labels = strings(numSamples, 1);   % String labels
    metadata = cell(numSamples, 1);    % Simulation metadata
    
    sampleIdx = 0;
    
    %% ===== GENERATE DATA FOR EACH FAULT CLASS =====
    tic;
    
    for f = 1:numel(faultTypes)
        fault = faultTypes(f);
        fprintf('Generating %d samples for fault: %s\n', nPerFault, fault);
        
        progressBar = '';
        
        for k = 1:nPerFault
            sampleIdx = sampleIdx + 1;
            
            % Progress indicator (every 20 samples)
            if mod(k, 20) == 0
                progressBar = [progressBar '▓'];
                fprintf('\r  Progress: %s %d/%d', progressBar, k, nPerFault);
            end
            
            % Sample system parameters with controlled variation
            params = sampleSystemParameters(config, k, nPerFault);
            
            % Choose valve command profile (varied patterns)
            valveProfile = selectValveProfile(k, nPerFault);
            
            % Simulate this scenario
            [sigFull, meta] = simulatePneumaticSystem(...
                config.t, config.Ts, params, fault, valveProfile);
            
            % Extract feature window (steady-state region)
            idxStart = config.N - config.WINDOW + 1;
            sig = sigFull(idxStart:end, :);  % WINDOW x numFeatures
            
            % Quality validation (Strict Check)
            if any(isnan(sig(:))) || any(isinf(sig(:))) || max(abs(sig(:))) > 1e9
                warning('Unstable simulation detected in sample %d. Regenerating parameters...', sampleIdx);
                k = k - 1;  % Retry this sample
                sampleIdx = sampleIdx - 1;
                continue;
            end
            
            % Store data
            X{sampleIdx} = sig.';   % Transpose to numFeatures x timeSteps
            labels(sampleIdx) = fault;
            metadata{sampleIdx} = meta;
        end
        
        fprintf('\n');
    end
    
    elapsedTime = toc;
    fprintf('\nDataset generation completed in %.2f seconds\n', elapsedTime);
    
    %% ===== CONVERT LABELS TO CATEGORICAL =====
    Y = categorical(labels);
    
    %% ===== DATASET STATISTICS =====
    fprintf('\n========================================\n');
    fprintf('DATASET STATISTICS\n');
    fprintf('========================================\n');
    
    % Class distribution
    fprintf('\nClass Distribution:\n');
    classCounts = countcats(Y);
    classNames = categories(Y);
    for i = 1:numel(classNames)
        fprintf('  %-20s: %4d samples (%.1f%%)\n', ...
            classNames{i}, classCounts(i), 100*classCounts(i)/numSamples);
    end
    
    % Feature statistics
    allData = cell2mat(X);
    fprintf('\nFeature Statistics (across all samples):\n');
    featureNames = {'P_measured', 'P_twin', 'Residual', 'Valve_cmd', ...
                    'dP/dt_meas', 'dP/dt_twin', 'Residual_MA'};
    numFeatures = numel(featureNames);
    for i = 1:numFeatures
        fprintf('  %-15s: mean=%8.2e, std=%8.2e, range=[%8.2e, %8.2e]\n', ...
            featureNames{i}, ...
            mean(allData(i,:)), ...
            std(allData(i,:)), ...
            min(allData(i,:)), ...
            max(allData(i,:)));
    end
    
    %% ===== SAVE DATASET =====
    datasetInfo = struct();
    datasetInfo.config = config;
    datasetInfo.faultTypes = faultTypes;
    datasetInfo.nPerFault = nPerFault;
    datasetInfo.numSamples = numSamples;
    datasetInfo.featureNames = featureNames;
    datasetInfo.generationDate = datetime('now');
    datasetInfo.generationTime = elapsedTime;
    
    save('pneumatic_dataset.mat', 'X', 'Y', 'metadata', 'datasetInfo', '-v7.3');
    
    fprintf('\n✓ Dataset saved to: pneumatic_dataset.mat\n');
    fprintf('  File size: %.2f MB\n', getFileSize('pneumatic_dataset.mat')/1e6);
    fprintf('\n========================================\n\n');
end

%% ===== HELPER FUNCTIONS =====
function params = sampleSystemParameters(config, sampleIdx, totalSamples)
%SAMPLESYSTEMPARAMETERS Generate system parameters with controlled variation
    
    % Base parameters (realistic pneumatic system)
    params.Ps = 7.0e5;              % Supply pressure 7 bar (gauge)
    params.Patm = 1.01325e5;        % Atmospheric pressure
    
    % FIXED: Capacitance increased slightly to ensure physical stability relative to flow
    % Or rely on sub-stepping. Let's keep small capacitance but handle it in ODE.
    params.C = 1.5e-7;              % Volume capacitance (Increased for stability 1e-9 -> 1e-7)
    
    params.R_in_nominal = 5e5;      % Inlet flow resistance
    params.R_leak_nominal = 3e6;    % Leak resistance (nominal)
    params.P0 = params.Patm;        % Initial pressure
    
    % Sensor parameters
    params.sensorNoise = 800;       % Gaussian noise std dev [Pa]
    params.sensorQuantization = 100;% ADC quantization [Pa]
    params.sensorBias_pos = 4e4;    % Positive bias [Pa]
    params.sensorBias_neg = -3.5e4; % Negative bias [Pa]
    
    % Add controlled variation (10% deviation from nominal)
    variationFactor = 0.10;
    progress = sampleIdx / totalSamples;  % 0 to 1
    
    % Smooth variation across dataset (not pure random)
    params.Ps = params.Ps * (1 + variationFactor * sin(2*pi*progress));
    params.C = params.C * (1 + variationFactor * cos(4*pi*progress));
    params.R_in_nominal = params.R_in_nominal * (1 + 0.05*randn());
    
    % Temperature effects (simplified)
    params.temperature = 20 + 10*rand();  % 20-30°C
    params.tempCoeff = 1 + 0.002*(params.temperature - 25);  % 0.2%/°C
end

function profile = selectValveProfile(sampleIdx, totalSamples)
%SELECTVALVEPROFILE Choose valve command pattern for diversity
    profileTypes = {'step4', 'step3', 'ramp', 'sine', 'mixed'};
    idx = mod(sampleIdx - 1, numel(profileTypes)) + 1;
    profile = profileTypes{idx};
end

function [sig, meta] = simulatePneumaticSystem(t, Ts, params, fault, valveProfile)
%SIMULATEPNEUMATICSYSTEM Main physics simulation with fault injection
    
    N = numel(t);
    
    % Preallocate state vectors
    P_real = zeros(N, 1);
    P_twin = zeros(N, 1);
    P_meas = zeros(N, 1);
    u = zeros(N, 1);
    
    % Initial conditions
    P_real(1) = params.P0;
    P_twin(1) = params.P0;
    
    %% ===== FAULT CONFIGURATION =====
    [leakFactor, valveStuck, valveStuckValue, sensorBias] = ...
        configureFault(fault, params);
    
    %% ===== VALVE COMMAND GENERATION =====
    u = generateValveCommand(t, valveProfile);
    
    %% ===== PHYSICS SIMULATION (WITH SUB-STEPPING) =====
    % CRITICAL FIX: The time constant is approx R*C (5e5 * 1.5e-7 = 0.075s).
    % Ideally safe. But if R drops (large u), dynamics get fast.
    % We use sub-stepping to guarantee stability.
    
    subSteps = 20; % 20 sub-steps per sample time
    dt_sub = Ts / subSteps;
    
    for k = 1:N-1
        % --- 1. Real Plant Simulation (Sub-stepped) ---
        p_curr = P_real(k);
        u_k = u(k);
        
        if valveStuck
            u_eff = valveStuckValue;
        else
            u_eff = u_k;
        end
        
        R_leak_real = params.R_leak_nominal * leakFactor * params.tempCoeff;
        
        for s = 1:subSteps
            p_curr = rungeKutta4(@(P) pneumaticODE(P, u_eff, params.Ps, ...
                params.C, params.R_in_nominal, R_leak_real), p_curr, dt_sub);
            % Physics Clamp: Pressure cannot exceed Ps + margin or go below Patm
            p_curr = max(params.Patm, min(p_curr, params.Ps * 1.5));
        end
        P_real(k+1) = p_curr;
        
        % --- 2. Digital Twin Simulation (Sub-stepped) ---
        p_twin_curr = P_twin(k);
        R_leak_twin = params.R_leak_nominal;
        
        for s = 1:subSteps
            p_twin_curr = rungeKutta4(@(P) pneumaticODE(P, u_k, params.Ps, ...
                params.C, params.R_in_nominal, R_leak_twin), p_twin_curr, dt_sub);
            p_twin_curr = max(params.Patm, min(p_twin_curr, params.Ps * 1.5));
        end
        P_twin(k+1) = p_twin_curr;
    end
    
    %% ===== SENSOR MEASUREMENT MODEL =====
    % Gaussian noise
    noise = params.sensorNoise * randn(N, 1);
    
    % Quantization (ADC effect)
    P_meas = P_real + sensorBias + noise;
    P_meas = round(P_meas / params.sensorQuantization) * params.sensorQuantization;
    
    %% ===== RESIDUAL COMPUTATION & FEATURE ENGINEERING =====
    
    % Force consistent signal lengths
    P_meas = P_meas(:);
    P_twin = P_twin(:);
    u      = u(:);
    
    residual = P_meas - P_twin;
    
    % CRITICAL FIX: Smooth data before derivative
    % Derivatives on quantized noise cause massive spikes (exploding gradients)
    P_meas_smooth = smoothdata(P_meas, 'gaussian', 10);
    P_twin_smooth = smoothdata(P_twin, 'gaussian', 10);
    
    dP_meas  = [0; diff(P_meas_smooth)] / Ts;
    dP_twin  = [0; diff(P_twin_smooth)] / Ts;
    
    % Ensure exact length match
    Nsig = numel(P_meas);
    dP_meas = dP_meas(1:Nsig);
    dP_twin = dP_twin(1:Nsig);
    
    windowSize = 20; % Increased window for smoother residual tracking
    residual_MA = movmean(residual, windowSize);
    residual_MA = residual_MA(:);
    
    % Final feature assembly
    sig = [ ...
        P_meas(1:Nsig), ...
        P_twin(1:Nsig), ...
        residual(1:Nsig), ...
        u(1:Nsig), ...
        dP_meas(1:Nsig), ...
        dP_twin(1:Nsig), ...
        residual_MA(1:Nsig) ...
    ];
    
    %% ===== METADATA =====
    meta = struct();
    meta.fault = fault;
    meta.valveProfile = valveProfile;
    meta.leakFactor = leakFactor;
    meta.valveStuck = valveStuck;
    meta.sensorBias = sensorBias;
    meta.finalPressure_real = P_real(end);
    meta.finalPressure_twin = P_twin(end);
    meta.meanResidual = mean(residual);
    meta.stdResidual = std(residual);
end

function dPdt = pneumaticODE(P, u, Ps, C, R_in, R_leak)
%PNEUMATICODE Differential equation for pneumatic system
    
    % Clamp valve command to prevent divide-by-zero
    u = max(min(u, 1.0), 1e-4);
    
    % Inlet flow (supply to tank)
    Rin_eff = R_in / u;
    Q_in = (Ps - P) / Rin_eff;
    
    % Leak flow (tank to atmosphere)
    Q_leak = P / R_leak;
    
    % Net rate of pressure change
    dPdt = (Q_in - Q_leak) / C;
end

function Pnext = rungeKutta4(f, P, h)
%RUNGEKUTTA4 4th-order Runge-Kutta integration
    k1 = f(P);
    k2 = f(P + 0.5*h*k1);
    k3 = f(P + 0.5*h*k2);
    k4 = f(P + h*k3);
    Pnext = P + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
end

function [leakFactor, valveStuck, valveStuckValue, sensorBias] = ...
    configureFault(fault, params)
%CONFIGUREFAULT Define fault parameters for each class
    
    % Defaults (normal operation)
    leakFactor = 1.0;
    valveStuck = false;
    valveStuckValue = 0.5;
    sensorBias = 0.0;
    
    switch fault
        case "normal"
            % All nominal
        case "leak_small"
            leakFactor = 0.40;  % 2.5x normal leak
        case "leak_large"
            leakFactor = 0.12;  % 8x normal leak
        case "leak_critical"
            leakFactor = 0.05;  % 20x normal leak
        case "valve_stuck_half"
            valveStuck = true;
            valveStuckValue = 0.5;
        case "valve_stuck_closed"
            valveStuck = true;
            valveStuckValue = 0.05;  % Nearly closed
        case "sensor_bias_pos"
            sensorBias = params.sensorBias_pos;
        case "sensor_bias_neg"
            sensorBias = params.sensorBias_neg;
        otherwise
            warning('Unknown fault type: %s', fault);
    end
end

function u = generateValveCommand(t, profile)
%GENERATEVALVECOMMAND Create diverse valve command patterns
    N = numel(t);
    u = zeros(N, 1);
    
    switch profile
        case 'step4'
            changeTimes = [0, 5, 10, 15];
            values = [0.3, 0.8, 0.5, 0.9];
            u = piecewiseConstant(t, changeTimes, values);
        case 'step3'
            changeTimes = [0, 7, 14];
            values = [0.2, 0.7, 0.4];
            u = piecewiseConstant(t, changeTimes, values);
        case 'ramp'
            u = 0.2 + 0.6 * (t / max(t));
        case 'sine'
            u = 0.5 + 0.3 * sin(2*pi*0.1*t);
        case 'mixed'
            u_base = piecewiseConstant(t, [0, 10], [0.4, 0.7]);
            u = u_base + 0.1 * sin(2*pi*0.2*t);
            u = max(min(u, 1.0), 0.1);
        otherwise
            u(:) = 0.5;
    end
    u = max(min(u, 1.0), 0.05);
    u = u(:);
end

function u = piecewiseConstant(t, changeTimes, values)
%PIECEWISECONSTANT Generate piecewise constant signal
    N = numel(t);
    u = zeros(N, 1);
    idx = 1;
    for k = 1:N
        if idx < numel(changeTimes) && t(k) >= changeTimes(idx+1)
            idx = idx + 1;
        end
        u(k) = values(idx);
    end
end

function fileSize = getFileSize(filename)
%GETFILESIZE Get file size in bytes
    info = dir(filename);
    if isempty(info)
        fileSize = 0;
    else
        fileSize = info.bytes;
    end
end
