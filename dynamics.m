syms x y theta real
syms px py real % [px, py] is locaion of contact in body frame (py is deviation from center).
syms vn vt real % [vn, vt] is velocity of contact in body frame.
syms mu c

state = [x; y; theta; py];
u = [vn; vt];

% box_half_width = 1;
% f_max = 1;
% m_max = 1;
% mu = 1;
% c = f_max / m_max;
constants = [mu, c, px];

% A1 = ccode(simplify(jacobian(f1(state, u, constants), state)));
% A2 = ccode(simplify(jacobian(f2(state, u, constants), state)));
% A3 = ccode(simplify(jacobian(f3(state, u, constants), state)));
% 
% B1 = ccode(simplify(jacobian(f1(state, u, constants), u)));
% B2 = ccode(simplify(jacobian(f2(state, u, constants), u)));
% B3 = ccode(simplify(jacobian(f3(state, u, constants), u)));

gt = (mu*c*c - px*py + mu*px*px) / (c*c + py*py - mu*px*py);
gb = (-mu*c*c - px*py - mu*px*px) / (c*c + py*py + mu*px*py);

Ct = ccode(simplify(jacobian(gt, state)));
Cb = ccode(simplify(jacobian(gb, state)));

function dx = f1(state, u, constants)
    x = state(1); y = state(2); theta = state(3); py = state(4);
    mu = constants(1); c = constants(2); px = constants(3);
    
    P1 = [1, 0; 0, 1];
    b1 = [-py / (c*c + px*px + py*py), px];
    c1 = [0, 0];
    
    dx = f(state, u, constants, P1, b1, c1);
end

function dx = f2(state, u, constants)
    x = state(1); y = state(2); theta = state(3); py = state(4);
    mu = constants(1); c = constants(2); px = constants(3);
    
    gt = (mu*c*c - px*py + mu*px*px) / (c*c + py*py - mu*px*py);
    
    P2 = [1, 0; gt, 0];
    b2 = [(-py + gt*px) / (c*c + px*px + py*py), 0];
    c2 = [-gt, 0];
    
    dx = f(state, u, constants, P2, b2, c2);
end

function dx = f3(state, u, constants)
    x = state(1); y = state(2); theta = state(3); py = state(4);
    mu = constants(1); c = constants(2); px = constants(3);
    
    gb = (-mu*c*c - px*py - mu*px*px) / (c*c + py*py + mu*px*py);
    
    P3 = [1, 0; gb, 0];
    b3 = [(-py + gb*px) / (c*c + px*px + py*py), 0];
    c3 = [-gb, 0];
    
    dx = f(state, u, constants, P3, b3, c3);
end

function dx = f(state, u, constants, P, b_vec, c_vec)
    x = state(1); y = state(2); theta = state(3); py = state(4);
    mu = constants(1); c = constants(2); px = constants(3);
    
%     gt = (mu*c*c - px*py + mu*px*px) / (c*c + py*py - mu*px*py);
%     gb = (-mu*c*c - px*py - mu*px*px) / (c*c + py*py + mu*px*py);
    
    C = [cos(theta), sin(theta); -sin(theta), cos(theta)];
    Q = (1 / (c*c + px*px + py*py)) * [c*c + px*px, px*py; px*py, c*c + py*py];
%     disp([C.' * Q * P; b_vec; c_vec])
    dx = [C.' * Q * P; b_vec; c_vec] * u;
end

function K = K1(state, u, constants)
    x = state(1); y = state(2); theta = state(3); py = state(4);
    mu = constants(1); c = constants(2); px = constants(3);
    C = [cos(theta), sin(theta); -sin(theta), cos(theta)];
    Q = (1 / (c*c + px*px + py*py)) * [c*c + px*px, px*py; px*py, c*c + py*py];
    P1 = [1, 0; 0, 1];
    
end


