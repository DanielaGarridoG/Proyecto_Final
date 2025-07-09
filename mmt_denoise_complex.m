function I_mmt = mmt_denoise_complex(I, J)
    if nargin < 2, J = 3; end
    I_mmt_real = real(I);
    I_mmt_imag = imag(I);
    for j = 1:J
        I_mmt_real = medfilt2(I_mmt_real, [3 3], 'symmetric');
        I_mmt_imag = medfilt2(I_mmt_imag, [3 3], 'symmetric');
    end
    I_mmt = I_mmt_real + 1i * I_mmt_imag;
end
