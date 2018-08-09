%% Initialization

% Reading the original image
I_skew = imread('logo.png');

% Camera position in front of the door, skewed view point to the logo
view_skew = [-40 0 25]'; 

% Camera position on top of the logo looking downward
view_up = [0 0 view_skew(3)]';

% Focal Length of the camera
focal_length_y = 500;

% Size of the input image
Isize_skew = size(I_skew);

% Principal point of the input image assuming it is in the middle of the
% image
principal_point_skew = Isize_skew/2;



%% Finding 3D coordinates of corners of the input image on the ground

% Corners of the input image
corners_skew = [
    1 1;
    Isize_skew(1) 1;
    1 Isize_skew(2);
    Isize_skew(1) Isize_skew(2);
    ];

% Back-projecting corners by finding their coordinates in camera coordinate
% system
corners_skew(:,1) = corners_skew(:,1) - principal_point_skew(1); %?
corners_skew(:,2) = corners_skew(:,2) - principal_point_skew(2); %?
corners_skew(:,1) = -corners_skew(:,1);
corners_skew = [corners_skew focal_length_y*ones(4,1)];%?

% Translation and Rotation of the coordinate system of the first image with
% respect to the global coordinate system located on the ground
t_skew = view_skew; %?
% Z-axis of the camera coordinate system is perpendicular to the image 
% plane and aims at the origin of the global coordinate system on the ground
z_axis_skew = -view_skew;%? 
y_axis_skew = cross(z_axis_skew,[0,0,1]);
x_axis_skew = cross(y_axis_skew , z_axis_skew);%? 
z_axis_skew = z_axis_skew/norm(z_axis_skew);
y_axis_skew = y_axis_skew/norm(y_axis_skew);
x_axis_skew = x_axis_skew/norm(x_axis_skew);
R_skew = [x_axis_skew y_axis_skew z_axis_skew]'; % Rotation matrix

% Finding normal vector of the ground plane with respect to the camera
% coordinate system; also, obtaining the distance of the camera to the
% ground plane
n_skew = R_skew * [0 ;0; 1]; % normal vector ?
d_skew = view_skew(3); % distance

% 3D location of points on the ground with respect to the camera coordinate
% system
depth = -d_skew./sum((corners_skew.*repmat(n_skew',[4,1])),2); % depth of each point
corners3Dskew = repmat(depth , [1,3]) .*corners_skew ; % 3D locations  ?

%% Obtaining 3D coordinates of corners on the ground with respect to the global coordinate system
corners3D = R_skew^(-1)*corners3Dskew'; %?
corners3D = corners3D';
corners3D = corners3D+repmat(t_skew',[4 1]);



%% Projecting 3D points on the second image

% Translation vector ans rotation matrix of the second image
t_up = view_up; %?
z_axis_up = -view_up; %?
if norm(cross(z_axis_up,[0,0,1]))<0.001
    y_axis_up = [0 -1 0];
else
    y_axis_up = cross(z_axis_up,[0,0,1]);
end
x_axis_up = cross(y_axis_up,z_axis_up); %?
z_axis_up = z_axis_up/norm(z_axis_up);
y_axis_up = y_axis_up'/norm(y_axis_up);
x_axis_up = x_axis_up/norm(x_axis_up);
R_up = [x_axis_up y_axis_up z_axis_up]'; % Rotation matrix

% Finding 3D coordinates of corners with respect to the second image
corners3Dup = corners3D-repmat(t_up',[4 1]); %? 
corners3Dup = R_up*corners3Dup';
corners3Dup = corners3Dup';

% Estimating size of the second image to be big enough to contain all 
% pixels of the logo and put the projection of the origin of the global 
% coordinate system at the middle of the image
corners_up = corners3Dup./repmat(corners3Dup(:,3),[1 3])* focal_length_y; %?
temp = ceil(2*max(abs(corners_up)));
Isize_up = temp(1:2); % Size of the second image
principal_point_up = Isize_up/2; % Principal point of the second image



%% Back-projecting all empty pixels in the second image

% All pixels of the second image
[columns, rows] = meshgrid( 1:Isize_up(2) , 1:Isize_up(1) );
pixels_up = [rows(:) columns(:)]; % each row is the coordinates of one pixel

% Back projecting pixels
pixels_up(:,1) = pixels_up(:,1) - principal_point_up(1); %?
pixels_up(:,2) = pixels_up(:,2) - principal_point_up(2); %?
pixels_up(:,1) = -pixels_up(:,1);
pixels_up = [pixels_up focal_length_y*ones(size(pixels_up,1),1)]; %?

% Ground plane with respect to the second image
n_up = R_up*[0 0 1]';
d_up = t_up(3);

% Coordinates of pixels on the ground with respect to the second image
depth = -d_up./sum((pixels_up.*repmat(n_up',size(pixels_up,1),1)),2); %?
pixels3D_up = pixels_up.*repmat(depth,[1 3]);

% Coordinates of the pixels on the ground with respect to the global system
pixels3D = R_up^(-1)*pixels3D_up'; %?
pixels3D = pixels3D';
pixels3D = pixels3D+repmat(t_up',[size(pixels3D_up,1) 1]);

% Coordinates of the pixels on the ground with respect to the first image
pixels3D_skew = pixels3D-repmat(t_skew',[size(pixels3D,1) 1]);
pixels3D_skew = R_skew*pixels3D_skew';
pixels3D_skew = pixels3D_skew'; %?

% Projection of the pixels in the first image
pixels_skew = pixels3D_skew./repmat(pixels3D_skew(:,3),[1 3])*focal_length_y; %?
pixels_skew(:,1) = pixels_skew(:,1) - principal_point_skew(1); %?
pixels_skew(:,2) = pixels_skew(:,2) + principal_point_skew(2); %?
pixels_skew(:,1) = -pixels_skew(:,1);
pixels_skew = pixels_skew(:,1:2);



%% Output image synthesizing

% All pixels in the second image
[columns, rows] = meshgrid( 1:Isize_up(2) , 1:Isize_up(1) );
pixels_up = [rows(:) columns(:)];

% Ignoring pixels ourside of the first image
pixels_skew = round(pixels_skew);
temp = sum(pixels_skew<1,2);
temp = temp + (pixels_skew(:,1)>Isize_skew(1));
temp = temp + (pixels_skew(:,2)>Isize_skew(2));
temp = temp>0;
pixels_skew(temp,:) = [];
pixels_up(temp,:) = [];

% Synthesizing the output image
I_up = uint8(zeros(Isize_up(1),Isize_up(2),3));
for i = 1:size(pixels_up,1)
   I_up(pixels_up(i,1),pixels_up(i,2),:) = I_skew(pixels_skew(i,1),pixels_skew(i,2),:); 
end

imshow(I_up);
imwrite(I_up , 'hw1_q3.jpg');
