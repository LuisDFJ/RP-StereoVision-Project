R = readmatrix( "results\reference_model.csv" );
V = readmatrix( "results\stereo_results.csv" );
V = V(2:end,:);

figure;
hold on;
grid on
axis equal

plot3( R(:,1), R(:,2), R(:,3), ":b", "LineWidth", 2 )
plot3( V(:,1), V(:,2), V(:,3), "b", "LineWidth", 2 )

plot3( R(:,4), R(:,5), R(:,6), ":r", "LineWidth", 2 )
plot3( V(:,4), V(:,5), V(:,6), "r", "LineWidth", 2 )

plot3( R(:,7), R(:,8), R(:,9), ":g", "LineWidth", 2 )
plot3( V(:,7), V(:,8), V(:,9), "g", "LineWidth", 2 )

plot3( R(1,1), R(1,2), R(1,3), "ob", "MarkerSize", 10 )
plot3( R(end,1), R(end,2), R(end,3), "xb", "MarkerSize", 10 )

plot3( R(1,4), R(1,5), R(1,6), "or", "MarkerSize", 10 )
plot3( R(end,4), R(end,5), R(end,6), "xr", "MarkerSize", 10 )

plot3( R(1,7), R(1,8), R(1,9), "og", "MarkerSize", 10 )
plot3( R(end,7), R(end,8), R(end,9), "xg", "MarkerSize", 10 )

fontsize(gca, 13, 'points')

xlabel( "X axis", "Interpreter", "latex", FontSize=22 )
ylabel( "Y axis", "Interpreter", "latex", FontSize=22 )
zlabel( "Z axis", "Interpreter", "latex", FontSize=22 )



