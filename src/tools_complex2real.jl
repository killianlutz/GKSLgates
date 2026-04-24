" Convert Lindbladian matrices to equivalent REAL matrices (L ↦ [Re(L) -Im(L); Re(L) Im(L)])"
function real_lindbladians(qs::QuantumSystem)
    L = lindbladians(qs)
    real_lindbladians(L)
end

function real_lindbladians(L::Lindbladians)
    C = eltype(L.Lfree)
    nrows = size(L.Lfree, 1)
    eye = oftype(L.Lfree, one(real(C))I(nrows))
    tensormatrix(A) = kron(eye, equiv_realmatrix(A))  # returns real matrix f(A) acting on real column vectors
    
    rLf = tensormatrix(L.Lfree)
    rLc = map(tensormatrix, L.Lcontrol)
    T = typeof(rLf)
    Lindbladians{T}(rLf, rLc) # equivalent real-valued Lindbladians
end

" Component-wise real and imaginary parts of matrix "
function complex_to_real(M)
    # complex to real
    (real(M), imag(M))
end

" Complex matrix associated to a real matrix with twice as many rows as columns ([Re; Im] ↦ Re + i Im) "
function real_to_complexmatrix(M)
    # real to complex matrix
    mid = Int(round(size(M, 1) / 2)) 
    @views M[1:mid,1:mid] .+ im .* M[(mid+1):end,1:mid]
end

" Real matrix [Re -Im; Im Re] associated to Complex mat. A = Re + i Im (component-wise) "
function equiv_realmatrix(A, B)
    [A -B;
     B  A]
end

function equiv_realmatrix(M)
    equiv_realmatrix(complex_to_real(M)...)
end 

" Stacking real/imaginary parts into column vector "
function equiv_realvector(A, B)
    vec([A; B])
end

function equiv_realvector(X)
    equiv_realvector(complex_to_real(X)...)
end

" Given real vector, stacks real/imaginary parts so as to obtain real matrix (identified to a complex one) with twice as many rows as columns "
function realvec_to_realmat(x)
    # column vector to matrix
    dim = Int(round(sqrt(length(x) / 2)))
    reshape(x, 2 * dim, dim)
end