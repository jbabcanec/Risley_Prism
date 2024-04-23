

%initial coordinate store
coordx{1} = [rx_{1};0;0];
coordy{1} = [0;ry_{1};0];
coordz{1} = [0;0;0];



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