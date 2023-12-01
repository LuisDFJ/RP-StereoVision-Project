clear all
s = load( "params\params_15mm_near_2.mat" );
params = s.stereoParams;
clear s

R   = params.PoseCamera2.R;
T   = params.PoseCamera2.Translation;

KL = params.CameraParameters1.K;
KR = params.CameraParameters2.K;

rdL = params.CameraParameters1.RadialDistortion;
rdR = params.CameraParameters2.RadialDistortion;

tdL = params.CameraParameters1.TangentialDistortion;
tdR = params.CameraParameters1.TangentialDistortion;

dL = [ rdL(1:2), tdL, rdL(3) ];
dR = [ rdR(1:2), tdR, rdR(3) ];

save_file_bin( "parameters\\R.bin", R )
save_file_bin( "parameters\\T.bin", T )
save_file_bin( "parameters\\KL.bin", KL )
save_file_bin( "parameters\\KR.bin", KR )
save_file_bin( "parameters\\dL.bin", dL )
save_file_bin( "parameters\\dR.bin", dR )

function save_file_bin( path, var )
f = fopen( path, "w");
fwrite( f, var, "double" );
fclose( f );
end