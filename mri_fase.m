clc; clear; close all;

%% 1. Función de unwrap Laplaciano personalizada
function unwrapped = laplacianUnwrap2D(phase, voxel_size)
    % Implementación alternativa del unwrap Laplaciano para 2D
    if nargin < 2
        voxel_size = [1 1];
    end
    
    phase = double(phase);
    psi = exp(1i*phase);
    laplacian_psi = del2(psi);
    
    kernel = [0 -1 0; -1 4 -1; 0 -1 0];
    inv_kernel = fft2(kernel, size(phase,1), size(phase,2));
    inv_kernel(1,1) = 1;
    
    rhs = imag(conj(psi).*laplacian_psi);
    unwrapped = real(ifft2(fft2(rhs)./inv_kernel));
end

%% 2. Función de unwrap 2D personalizada
function unwrapped = myUnwrap2D(phase)
    unwrapped = zeros(size(phase));
    for i = 1:size(phase,1)
        unwrapped(i,:) = unwrap(phase(i,:));
    end
    for j = 1:size(phase,2)
        unwrapped(:,j) = unwrap(unwrapped(:,j));
    end
end

%% 3. Configuración inicial
try
    vol = niftiread('brain.nii');
    slice = double(vol(:,:,100));
    mag = slice / max(slice(:));
catch
    mag = phantom(256);
    fprintf('Usando datos simulados (phantom)\n');
end

mask = mag > 0.05;
[X,Y] = meshgrid(1:size(mag,2), 1:size(mag,1));
phi_true = 0.1*X + 0.1*Y + 0.001*(X-size(mag,2)/2).^2;
phi_ref = myUnwrap2D(phi_true);

%% 4. Crear datos con ruido
I_clean = mag .* exp(1i*phi_true);
I_noisy = I_clean + 0.05*(randn(size(I_clean)) + 1i*randn(size(I_clean)));

%% 5. Preprocesamiento
fprintf('Aplicando filtros...\n');
tic;
I_median = medfilt2(real(I_noisy),[3 3]) + 1i*medfilt2(imag(I_noisy),[3 3]);
time_median = toc;

tic;
I_mmt = I_noisy; % Reemplaza con tu filtro MMT real
time_mmt = toc;

%% 6. Configuración de métodos
method_names = {'SUNWRAP', 'Laplaciano', 'Laplaciano Iter'};
method_funcs = {
    @(x) sunwrap(x, 0.05),
    @(x) laplacianUnwrap2D(angle(x), [1 1]),
    @(x) laplacianUnwrap2D(angle(x), [1 1]) + 2*pi*round((myUnwrap2D(angle(x))-laplacianUnwrap2D(angle(x), [1 1]))/(2*pi))
};

input_names = {'Sin filtro', 'Mediana', 'MMT'};
input_data = {I_noisy, I_median, I_mmt};

%% 7. Procesamiento principal
results = cell(length(input_names), length(method_names));

for i = 1:length(input_names)
    fprintf('\nEntrada: %s\n', input_names{i});
    for j = 1:length(method_names)
        fprintf(' - Método: %-15s', method_names{j});
        tic;
        
        phi_unw = method_funcs{j}(input_data{i});
        phi_unw = phi_unw + mean(phi_ref(mask) - phi_unw(mask));
        
        diff = phi_unw(mask) - phi_ref(mask);
        mse = mean(diff.^2);
        psnr = 10*log10(max(phi_ref(:))^2/mse);
        residues = sum(abs(round(diff/(2*pi))));
        
        results{i,j} = struct(...
            'time', toc, ...
            'psnr', psnr, ...
            'residues', residues, ...
            'unwrapped', phi_unw);
            
        fprintf('Tiempo: %.2fs | PSNR: %.1fdB | Residuos: %d\n', ...
                results{i,j}.time, results{i,j}.psnr, results{i,j}.residues);
    end
end

%% 8. Visualización de resultados
figure('Position',[100 100 1200 600]);
for i = 1:length(input_names)
    for j = 1:length(method_names)
        subplot(length(input_names), length(method_names), (i-1)*length(method_names)+j);
        imagesc(results{i,j}.unwrapped);
        title(sprintf('%s + %s\nPSNR: %.1f dB', input_names{i}, method_names{j}, results{i,j}.psnr));
        axis image off; colorbar;
    end
end

%% 9. Gráfico de tiempos (Versión corregida)
figure;
times = zeros(length(input_names), length(method_names));
for i = 1:length(input_names)
    for j = 1:length(method_names)
        times(i,j) = results{i,j}.time;
    end
end

bar(times');
set(gca, 'XTickLabel', method_names, 'FontSize', 10);
ylabel('Tiempo (s)');
title('Tiempo de ejecución por método');
legend(input_names, 'Location', 'northwest'); % Corrección aplicada aquí
grid on;