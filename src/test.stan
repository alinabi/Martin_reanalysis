
    data {
        int<lower=1> N;
        int<lower=0> t[N];
        vector[N] w;
    }
    
    transformed data {
        int<lower=N> T = max(t);
	int<lower=2, upper=T+1> present_t[N]; 
        int<lower=2, upper=T+1> missing_t[T - N];
        
        {
            int t_curs = 1;
            int m_curs = 1;
            
            for (j in 1:T) {
                if (j == t[t_curs]) {
		    present_t[t_curs] = j + 1;
                    t_curs += 1;
                }
                else {
                    missing_t[m_curs] = j + 1;
                    m_curs += 1;
                }
            }  
        }
    }
    
    parameters {
        real mu;
        real<lower=0> sigma;
        vector[T - N] xi;
    }
    
    transformed parameters {
        vector[T + 1] zeta;
        
        zeta[1] = 0;
        zeta[present_t] = w;
        zeta[missing_t] = xi;     
    }
    
    model {

        target += normal_lpdf(zeta[2:] - zeta[:T] | mu, sigma);
    }
