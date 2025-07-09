%% Código para comparación de métodos de phase unwrapping con datos reales de MRI
% Ahora incluye adición de gradientes suaves para simular wraps como sugiere el profesor

%% 1. Cargar y preparar los datos
clear; close all; clc;

% Cargar los archivos .mat
load('phs_tissue.mat'); % Contiene phs_tissue (160x160x160 double)
load('magn.mat');       % Contiene magn (160x160x160 single)

% Seleccionar una rebanada 2D (usar la del medio como ejemplo)
num_rebanada = 80;
magn_2d = double(magn(:,:,num_rebanada)); % Convertir a double
phs_2d = phs_tissue(:,:,num_rebanada);

%% 2. Agregar gradiente suave para generar wraps (como sugirió el profesor)
[X, Y] = meshgrid(1:size(magn_2d,2), 1:size(magn_2d,1));

% Crear gradiente polinómico (ajustar coeficientes según necesidad)
gradient_field = 0.05*X + 0.03*Y + 0.001*(X-size(magn_2d,2)/2).^2;

% Añadir gradiente a la fase local y envolver a [-pi, pi]
phs_wrapped = angle(exp(1i*(phs_2d + gradient_field)));

% Crear imagen compleja con la nueva fase envuelta
imagen_compleja = magn_2d .* exp(1i * phs_wrapped);

% Tamaño de voxel
voxel_size = [1 1]; % [ancho alto] en mm

%% 3. Agregar ruido gaussiano (opcional)
nivel_ruido = 0.05;
ruido_real = randn(size(imagen_compleja)) * nivel_ruido;
ruido_imag = randn(size(imagen_compleja)) * nivel_ruido;
imagen_ruidosa = imagen_compleja + ruido_real + 1i*ruido_imag;

%% 4. Filtrado de la imagen
disp('Aplicando filtros...');

% Filtro de mediana
tic;
parte_real = real(imagen_ruidosa);
parte_imag = imag(imagen_ruidosa);
imagen_mediana = medfilt2(parte_real, [3 3]) + 1i*medfilt2(parte_imag, [3 3]);
tiempo_mediana = toc;

% Filtro MMT
tic;
imagen_mmt = mmt_denoise_complex(imagen_ruidosa, 3);
tiempo_mmt = toc;

%% 5. Configuración de métodos de phase unwrapping
method_names = {'SUNWRAP', 'Laplacian', 'Laplacian Iter'};
method_funcs = {
    @(x) sunwrap(x, 0.05),
    @(x) unwrapLaplacian(angle(x), size(x), voxel_size),
    @(x) unwrap_iterativo(angle(x), voxel_size)
};

input_names = {'Sin filtro', 'Mediana', 'MMT'};
input_data = {imagen_ruidosa, imagen_mediana, imagen_mmt};

%% 6. Procesamiento principal
results = cell(length(input_names), length(method_names));

for i = 1:length(input_names)
    fprintf('\nEntrada: %s\n', input_names{i});
    for j = 1:length(method_names)
        fprintf(' - Método: %-15s', method_names{j});
        tic;
        
        % Aplicar el método de phase unwrapping
        fase_unw = method_funcs{j}(input_data{i});
        
        % Calcular métricas usando el gradiente conocido como referencia
        diff = fase_unw - gradient_field;
        mse = mean(diff(magn_2d > 0.05).^2);
        psnr = 10*log10(max(gradient_field(:))^2/mse);
        residues = sum(abs(round(diff(magn_2d > 0.05)/(2*pi))));
        
        % Guardar resultados
        results{i,j} = struct(...
            'time', toc, ...
            'mse', mse, ...
            'psnr', psnr, ...
            'residues', residues, ...
            'unwrapped', fase_unw);
            
        fprintf('Tiempo: %.2fs | PSNR: %.1fdB | Residuos: %d\n', ...
                results{i,j}.time, results{i,j}.psnr, results{i,j}.residues);
    end
end

%% 7. Visualización de resultados

% 1. Figura 1: Datos originales
figure('Units','normalized','Position',[0.1 0.1 0.9 0.25]);
colormap('jet');

subplot(1,3,1);
imagesc(gradient_field);
title('Gradiente añadido');
axis image off; colorbar;

subplot(1,3,2);
imagesc(phs_wrapped);
title('Fase envuelta con gradiente');
axis image off; colorbar;

subplot(1,3,3);
imagesc(magn_2d);
title('Magnitud');
axis image off; colorbar;

% Guardar figura 1
print('figura1_datos_originales.png', '-dpng', '-r300');

% 2. Figura 2: Resultados SUNWRAP
figure('Units','normalized','Position',[0.1 0.1 0.9 0.25]);
colormap('jet');

subplot(1,3,1);
imagesc(results{1,1}.unwrapped);
title(sprintf('Sin filtro + SUNWRAP\nPSNR: %.1f dB', results{1,1}.psnr));
axis image off; colorbar;

subplot(1,3,2);
imagesc(results{2,1}.unwrapped);
title(sprintf('Mediana + SUNWRAP\nPSNR: %.1f dB', results{2,1}.psnr));
axis image off; colorbar;

subplot(1,3,3);
imagesc(results{3,1}.unwrapped);
title(sprintf('MMT + SUNWRAP\nPSNR: %.1f dB', results{3,1}.psnr));
axis image off; colorbar;

% Guardar figura 2
print('figura2_resultados_sunwrap.png', '-dpng', '-r300');

% 3. Figura 3: Resultados Laplacian
figure('Units','normalized','Position',[0.1 0.1 0.9 0.25]);
colormap('jet');

subplot(1,3,1);
imagesc(results{1,2}.unwrapped);
title(sprintf('Sin filtro + Laplacian\nPSNR: %.1f dB', results{1,2}.psnr));
axis image off; colorbar;

subplot(1,3,2);
imagesc(results{2,2}.unwrapped);
title(sprintf('Mediana + Laplacian\nPSNR: %.1f dB', results{2,2}.psnr));
axis image off; colorbar;

subplot(1,3,3);
imagesc(results{3,2}.unwrapped);
title(sprintf('MMT + Laplacian\nPSNR: %.1f dB', results{3,2}.psnr));
axis image off; colorbar;

% Guardar figura 3
print('figura3_resultados_laplacian.png', '-dpng', '-r300');

% 4. Figura 4: Resultados Laplacian Iterativo
figure('Units','normalized','Position',[0.1 0.1 0.9 0.25]);
colormap('jet');

subplot(1,3,1);
imagesc(results{1,3}.unwrapped);
title(sprintf('Sin filtro + Laplacian Iter\nPSNR: %.1f dB', results{1,3}.psnr));
axis image off; colorbar;

subplot(1,3,2);
imagesc(results{2,3}.unwrapped);
title(sprintf('Mediana + Laplacian Iter\nPSNR: %.1f dB', results{2,3}.psnr));
axis image off; colorbar;

subplot(1,3,3);
imagesc(results{3,3}.unwrapped);
title(sprintf('MMT + Laplacian Iter\nPSNR: %.1f dB', results{3,3}.psnr));
axis image off; colorbar;

% Guardar figura 4
print('figura4_resultados_laplacian_iter.png', '-dpng', '-r300');


%% 8. Gráfico de métricas comparativas
figure('Position',[100 100 1200 400]);

% Preparar datos para la tabla
metrics = {'Tiempo (s)'; 'PSNR (dB)'; 'Residuos'};
data = zeros(length(metrics), length(input_names)*length(method_names));

for i = 1:length(input_names)
    for j = 1:length(method_names)
        col = (i-1)*length(method_names)+j;
        data(1,col) = results{i,j}.time;
        data(2,col) = results{i,j}.psnr;
        data(3,col) = results{i,j}.residues;
    end
end

% Crear tabla
column_names = {};
for i = 1:length(input_names)
    for j = 1:length(method_names)
        column_names{end+1} = [input_names{i} ' + ' method_names{j}];
    end
end

% Mostrar tabla
uitable('Data', data, 'RowName', metrics, 'ColumnName', column_names, ...
    'Position', [20 20 1160 360]);

%% 9. Comparación de diferencias entre métodos
figure('Name','Diferencias entre métodos (MMT filtrado)');

% Obtener resultados con filtro MMT
fase_mmt_sunwrap = results{3,1}.unwrapped;
fase_mmt_laplacian = results{3,2}.unwrapped;
fase_mmt_iter = results{3,3}.unwrapped;

% Calcular diferencias
subplot(1,3,1);
imagesc(fase_mmt_sunwrap - fase_mmt_laplacian);
title('Diff: SUNWRAP vs Laplacian');
axis image off; colorbar; colormap('jet');

subplot(1,3,2);
imagesc(fase_mmt_sunwrap - fase_mmt_iter);
title('Diff: SUNWRAP vs Laplacian Iter');
axis image off; colorbar; colormap('jet');

subplot(1,3,3);
imagesc(fase_mmt_laplacian - fase_mmt_iter);
title('Diff: Laplacian vs Laplacian Iter');
axis image off; colorbar; colormap('jet');
