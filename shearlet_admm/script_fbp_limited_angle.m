function gen_lim_angle(src_dir, out_dir)
    % Simple script to generate FBP reconstruction for images from the src_dir to the out_dir

    if not(isfolder(src_dir))
        error('Given source directory does not exist!');
    end

    if not(isfolder(out_dir))
        mkdir(out_dir);
    end

    files = dir(src_dir);
    files(ismember({files.name}, {'.', '..'})) = [];  % remove . and ..

    theta = 0:180;
    missing_wedge = 45:80;


    for i = 1:length(files)
        image = files(i);
        I = imread(strcat(src_dir, '/', image.name));

        I = rgb2gray(I);

        [R, xp] = radon(I, theta);

        R(:, missing_wedge) = 0;

        IR = iradon(R, theta, 'linear','Ram-Lak');

        imwrite(IR, strcat(out_dir, '/', image.name));
    end
