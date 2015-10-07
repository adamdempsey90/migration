from numpy import zeros,ones,array,arange,diag,allclose
from numpy.linalg import solve
from numpy.random import rand
def trisolve(diags,d):
    a = diags[0]
    b = diags[1]
    c= diags[2]
    nx = len(d)

# Forward sweep

    cp = zeros(c.shape)
    bp = ones(b.shape)
    dp = zeros(d.shape)

    cp[0] = c[0]/b[0]
    for i in range(1,nx-1):
        cp[i] = c[i]/(b[i]-a[i-1]*cp[i-1])

    dp[0] = d[0]/b[0]

    for i in range(1,nx):
        dp[i] = (d[i] - a[i-1]*dp[i-1])/(b[i]-a[i-1]*cp[i-1])

    sol = zeros(d.shape)

    sol[-1] = dp[-1]

    for i in arange(nx-1)[::-1]:
        sol[i] = dp[i] - cp[i]*sol[i+1]

    return sol


def generate_random(nx):
    md=rand(nx)
    ud=rand(nx-1)
    ld=rand(nx-1)
    A = diag(md,0)+diag(ud,1)+diag(ld,-1)
    b = rand(nx)


    return ld,md,ud,b

def test_case(nx):
    md=rand(nx)
    ud=rand(nx-1)
    ld=rand(nx-1)
    A = diag(md,0)+diag(ud,1)+diag(ld,-1)
    b = rand(nx)

    y = solve(A,b)
    yp = trisolve((ld,md,ud),b)

    err = abs((y-yp)/y)
    print 'Max Relative Error: %.2e' % err.max()
    print 'All close test = ' + str(allclose(y,yp))
