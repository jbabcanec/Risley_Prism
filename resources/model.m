%#10 Spinning Model 3D Timestep w/ Vectors
clear, clc
format longg

wedgenum = input('# of interfaces: ');
timelim = input('input amount of time (sec): ');
inc = input('enter how many time-steps for the amount of time: ');
time = linspace(0,timelim,inc);
timelength = length(time);
%N rotation
for i=1:wedgenum
N_{i} = input(['enter rotations/sec for wedge ' num2str(i) ' : ']);
end


thetax_{1} = input('enter initial laser angle with respect to x: ');
thetay_{1} = input('enter initial laser angle with respect to y: ');
rx_{1} = input('enter initial laser height with respect to x: ');
ry_{1} = input('enter initial laser height with respect to y: ');
d = input('enter wedge diameter: ');
%initial coordinate store
coordx{1} = [rx_{1};0;0];
coordy{1} = [0;ry_{1};0];
%phi
for i=1:wedgenum
startphix_{i} = input(['enter phi for interface ' num2str(i) ': ']);
end
%k
for i=1:wedgenum
k{i} = input(['enter optical axis distance between interface ' num2str(i-1) ' and ' num2str(i) ': ']);
end
k{wedgenum+1} = input('enter distance from last wedge to workpiece: ');
%n_j
for i=1:wedgenum+1
n_{i} = input(['enter n_' num2str(i) ' before interface ' num2str(i) ': ']);
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%Iterate Starts HERE
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%


for iter=1:length(time)
	for i=1:wedgenum
		phix_{i} = startphix_{i};
	end
	
	%gamma
	for i=1:wedgenum
		gamma_{i} = mod(360*N_{i}*time(iter),360);
	end

	%modified phi's
	for i=1:wedgenum

		%vectors
		n1 = [cosd(gamma_{i})*tand(phix_{i}); sind(gamma_{i})*tand(phix_{i}); -1];
		ny = [0; 1; 0];
		nx = [1; 0; 0];

		%angle
		phix_{i} = 90 - acosd((dot(n1,nx))/(norm(nx)*norm(n1)));
		phiy_{i} = 90 - acosd((dot(n1,ny))/(norm(ny)*norm(n1)));
	end

	%generate material equations and plot them
	sumk = 0;

	for i=1:wedgenum
		sumk = sumk + k{i};
		K{i} = sumk;
	end

	%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
	%X calcuations
	%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

	%initial laser path x
	x1 = rx_{1};
	x2 = rx_{1} + tand(thetax_{1});
	x3 = 0;
	z1 = 0;
	z2 = 1;
	z3 = k{1};

	if phix_{1} == 0
		x4 = 1;%arbitrary
		z4 = k{1};
	else
		x4 = cotd(phix_{1});
		z4 = k{1} + 1;
	end

	%intersection equations
	Px_{1} = ((x1*z2 - z1*x2)*(x3 - x4) - (x1 - x2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
	Pz_{1} = ((x1*z2 - z1*x2)*(z3 - z4) - (z1 - z2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
	coordx{2} = [Px_{1};0;Pz_{1}];

	%ending loop defaults
	K{wedgenum+1} = K{wedgenum} + k{wedgenum+1};
	phix_{wedgenum+1} = 0;
	gamma_{wedgenum+1} = 0;


	for i=1:wedgenum
		N = [tand(phix_{i});0;-1];
		s_i = [tand(thetax_{i});0;1];
		zmeasure = [0;0;1];

		N = N/norm(N);
		s_i = s_i/norm(s_i);

		%governing equation of refraction
		s_f = (n_{i}/n_{i+1})*(cross(N,cross(-N,s_i))) - N*((1 - ((n_{i}/n_{i+1}).^2)*dot(cross(N,s_i),cross(N,s_i))).^(1/2));
		
		%new output angle
		thetax_{i+1} = ((abs(s_f(1,1)))/s_f(1,1))*acosd(dot(zmeasure,s_f)/(norm(s_f)*norm(zmeasure)));

		%new r_i coordinates
		x1 = Px_{i};
		x2 = Px_{i} + tand(thetax_{i+1});
		x3 = 0;
		z1 = Pz_{i};
		z2 = Pz_{i} + 1;
		z3 = K{i+1};

		if phix_{i+1} == 0
			x4 = 1;%arbitrary
			z4 = K{i+1};
		else
			x4 = cotd(phix_{i+1});
			z4 = K{i+1} + 1;
		end

		Px_{i+1} = ((x1*z2 - z1*x2)*(x3 - x4) - (x1 - x2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
		Pz_{i+1} = ((x1*z2 - z1*x2)*(z3 - z4) - (z1 - z2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
	end

	for i=2:wedgenum+1
		coordx{i+1} = [Px_{i};0;Pz_{i}];
	end

	%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
	%Y calculations
	%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

	%initial laser path y
	y1 = ry_{1};
	y2 = ry_{1} + tand(thetay_{1});
	y3 = 0;
	z1 = 0;
	z2 = 1;
	z3 = k{1};

	if phiy_{1} == 0
		y4 = 1;%arbitrary
		z4 = k{1};
	else
		y4 = cotd(phiy_{1});
		z4 = k{1} + 1;
	end

	Py_{1} = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4))/((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4));
	Pz_{1} = ((y1*z2 - z1*y2)*(z3 - z4) - (z1 - z2)*(y3*z4 - z3*y4))/((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4));
	coordy{2} = [0;Py_{1};Pz_{1}];

	%ending loop defaults
	phiy_{wedgenum+1} = 0;
	
	for i=1:wedgenum
		N = [tand(phiy_{i});0;-1];
		s_i = [tand(thetay_{i});0;1];
		zmeasure = [0;0;1];
		N = N/norm(N);
		s_i = s_i/norm(s_i);

		%governing equation of refraction
		s_f = (n_{i}/n_{i+1})*(cross(N,cross(-N,s_i))) - N*((1 - ((n_{i}/n_{i+1}).^2)*dot(cross(N,s_i),cross(N,s_i))).^(1/2));
		
		%new output angle
		thetay_{i+1} = ((abs(s_f(1,1)))/s_f(1,1))*acosd(dot(zmeasure,s_f)/(norm(s_f)*norm(zmeasure)));

		%new r_i coordinates
		y1 = Py_{i};
		y2 = Py_{i} + tand(thetay_{i+1});
		y3 = 0;
		z1 = Pz_{i};
		z2 = Pz_{i} + 1;
		z3 = K{i+1};

		if phiy_{i+1} == 0
			y4 = 1;%arbitrary
			z4 = K{i+1};
		else
			y4 = cotd(phiy_{i+1});
			z4 = K{i+1} + 1;
		end

		Py_{i+1} = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4))/((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4));
		Pz_{i+1} = ((y1*z2 - z1*y2)*(z3 - z4) - (z1 - z2)*(y3*z4 - z3*y4))/((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4));
	end

	for i=2:wedgenum+1
		coordy{i+1} = [0;Py_{i};Pz_{i}];
	end

	%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
	%Z calculations
	%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

	coordz{1} = [0;0;0];

	for i=1:wedgenum+1
		coordz{i+1} = [0;0;K{i} + (((coordx{i+1}(1,1))*cosd(gamma_{i}) - (coordy{i+1}(2,1))*sind(gamma_{i}))*tand(phix_{i}))];
	end

	%laser coordinates x,y,z
	for i=1:wedgenum+2
		LaserCoord{i} = [coordx{i}(1,1);coordy{i}(2,1);coordz{i}(3,1)];
	end

	%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
	%Plotting
	%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

	scatter3(LaserCoord{wedgenum+2}(1,1),LaserCoord{wedgenum+2}(2,1),LaserCoord{wedgenum+2}(3,1),'filled')
	grid on
	hold on

end

pbaspect([1 1 1])