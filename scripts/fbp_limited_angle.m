src = ;
dest = ;

P = rgb2gray(imread(src));

theta = 0:359;
[R, xp] = radon(P, theta);

R(:, 60: 120) = 0;
R(:, 60 + 180: 120 + 180) = 0;

IR = iradon(R, theta, 'linear','Ram-Lak');

imwrite(uint8(IR), dest);
