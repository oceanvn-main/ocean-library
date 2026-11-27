function h = gen_hash(settings)
% GEN_HASH  8-hex-digit fingerprint of a settings struct (MD5 of its JSON).
%   Used in checkpoint names so that ANY change to a training setting
%   (widths, latents, learning rate, seed, ...) trains a new model instead of
%   reloading one trained with different settings.
md = java.security.MessageDigest.getInstance('MD5');
d = typecast(md.digest(uint8(jsonencode(settings))), 'uint8');
h = lower(reshape(dec2hex(d(1:4)).', 1, []));
end
