
% Validation script - identical to Python parameters
clear, clc
format longg

% Fixed parameters from Python
wedgenum = 3;
timelim = 5.0;
inc = 50;
time = linspace(0,timelim,inc);

% Rotation speeds
N_{1} = 1.0;
N_{2} = 0.5;
N_{3} = 1.5;

% Initial conditions
thetax_{1} = 10.0;
thetay_{1} = 5.0;
rx_{1} = 0.0;
ry_{1} = 0.0;
d = 10; % Standard diameter

% Initial coordinates
coordx{1} = [rx_{1};0;0];
coordy{1} = [0;ry_{1};0];

% Phi angles
startphix_{1} = 5.0;
startphix_{2} = 8.0;
startphix_{3} = 3.0;

% Distances
k{1} = 6.0;
k{2} = 6.0;
k{3} = 6.0;
k{4} = 6.0; % Distance to workpiece

% Refractive indices
n_{1} = 1.0;
n_{2} = 1.2;
n_{3} = 1.3;
n_{4} = 1.4;

% Main simulation loop
workpiece_coords = [];

for iter=1:length(time)
    for i=1:wedgenum
        phix_{i} = startphix_{i};
    end
    
    % Gamma calculations
    for i=1:wedgenum
        gamma_{i} = mod(360*N_{i}*time(iter),360);
    end
    
    % Modified phi's
    for i=1:wedgenum
        n1 = [cosd(gamma_{i})*tand(phix_{i}); sind(gamma_{i})*tand(phix_{i}); -1];
        ny = [0; 1; 0];
        nx = [1; 0; 0];
        
        phix_{i} = 90 - acosd((dot(n1,nx))/(norm(nx)*norm(n1)));
        phiy_{i} = 90 - acosd((dot(n1,ny))/(norm(ny)*norm(n1)));
    end
    
    % Calculate cumulative distances
    sumk = 0;
    for i=1:wedgenum
        sumk = sumk + k{i};
        K{i} = sumk;
    end
    
    % X calculations
    x1 = rx_{1};
    x2 = rx_{1} + tand(thetax_{1});
    x3 = 0;
    z1 = 0;
    z2 = 1;
    z3 = k{1};
    
    if phix_{1} == 0
        x4 = 1;
        z4 = k{1};
    else
        x4 = cotd(phix_{1});
        z4 = k{1} + 1;
    end
    
    Px_{1} = ((x1*z2 - z1*x2)*(x3 - x4) - (x1 - x2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
    Pz_{1} = ((x1*z2 - z1*x2)*(z3 - z4) - (z1 - z2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
    
    K{wedgenum+1} = K{wedgenum} + k{wedgenum+1};
    phix_{wedgenum+1} = 0;
    
    for i=1:wedgenum
        N = [tand(phix_{i});0;-1];
        s_i = [tand(thetax_{i});0;1];
        zmeasure = [0;0;1];
        
        N = N/norm(N);
        s_i = s_i/norm(s_i);
        
        s_f = (n_{i}/n_{i+1})*(cross(N,cross(-N,s_i))) - N*((1 - ((n_{i}/n_{i+1}).^2)*dot(cross(N,s_i),cross(N,s_i))).^(1/2));
        thetax_{i+1} = ((abs(s_f(1,1)))/s_f(1,1))*acosd(dot(zmeasure,s_f)/(norm(s_f)*norm(zmeasure)));
        
        x1 = Px_{i};
        x2 = Px_{i} + tand(thetax_{i+1});
        x3 = 0;
        z1 = Pz_{i};
        z2 = Pz_{i} + 1;
        z3 = K{i+1};
        
        if phix_{i+1} == 0
            x4 = 1;
            z4 = K{i+1};
        else
            x4 = cotd(phix_{i+1});
            z4 = K{i+1} + 1;
        end
        
        Px_{i+1} = ((x1*z2 - z1*x2)*(x3 - x4) - (x1 - x2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
        Pz_{i+1} = ((x1*z2 - z1*x2)*(z3 - z4) - (z1 - z2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
    end
    
    % Y calculations
    y1 = ry_{1};
    y2 = ry_{1} + tand(thetay_{1});
    y3 = 0;
    z1 = 0;
    z2 = 1;
    z3 = k{1};
    
    if phiy_{1} == 0
        y4 = 1;
        z4 = k{1};
    else
        y4 = cotd(phiy_{1});
        z4 = k{1} + 1;
    end
    
    Py_{1} = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4))/((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4));
    
    phiy_{wedgenum+1} = 0;
    
    for i=1:wedgenum
        N = [tand(phiy_{i});0;-1];
        s_i = [tand(thetay_{i});0;1];
        zmeasure = [0;0;1];
        N = N/norm(N);
        s_i = s_i/norm(s_i);
        
        s_f = (n_{i}/n_{i+1})*(cross(N,cross(-N,s_i))) - N*((1 - ((n_{i}/n_{i+1}).^2)*dot(cross(N,s_i),cross(N,s_i))).^(1/2));
        thetay_{i+1} = ((abs(s_f(1,1)))/s_f(1,1))*acosd(dot(zmeasure,s_f)/(norm(s_f)*norm(zmeasure)));
        
        y1 = Py_{i};
        y2 = Py_{i} + tand(thetay_{i+1});
        y3 = 0;
        z1 = Pz_{i};
        z2 = Pz_{i} + 1;
        z3 = K{i+1};
        
        if phiy_{i+1} == 0
            y4 = 1;
            z4 = K{i+1};
        else
            y4 = cotd(phiy_{i+1});
            z4 = K{i+1} + 1;
        end
        
        Py_{i+1} = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4))/((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4));
    end
    
    % Z calculations
    for i=1:wedgenum+1
        gamma_{i} = mod(360*N_{min(i,wedgenum)}*time(iter),360);
        if i <= wedgenum
            coordz_val = K{i} + (Px_{i}*cosd(gamma_{i}) - Py_{i}*sind(gamma_{i}))*tand(phix_{i});
        else
            coordz_val = K{i};
        end
    end
    
    % Store final workpiece coordinate
    workpiece_coords = [workpiece_coords; Px_{wedgenum+1}, Py_{wedgenum+1}, K{wedgenum+1}];
end

% Save results
save('matlab_validation_results.mat', 'workpiece_coords', 'time');
fprintf('MATLAB validation complete. Results saved to matlab_validation_results.mat\n');
fprintf('Final workpiece coordinates shape: %dx%d\n', size(workpiece_coords));
