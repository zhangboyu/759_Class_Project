figure();
axis equal;
axis([0 60 0 60]);
set(gca, 'Xtick', 0:12:60);
set(gca, 'Ytick', 0:12:60);
grid on;
circle(9,9,4);
circle(15, 8, 3.5);
circle(20, 9, 3);
circle(6, 18, 3);
circle(15, 19, 4);

function h = circle(x,y,r)
    hold on
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit);
    hold off 
end