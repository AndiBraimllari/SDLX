P = phantom(512);

% imshow(P);

theta = 0:180;
[R,xp] = radon(P, theta);

R(:, 45:135) = 0;

IR = iradon(R, theta, 'linear','Ram-Lak');

imshow(IR);
