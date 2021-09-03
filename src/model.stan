functions {
    vector hazard_modulation(vector t, real t_min, real eta) {
	vector[size(t)] res = 0.5 * (1 + tanh((t - t_min) / eta));
	return res;
    }
}

data {
    /* steepness of sigmoid hazard modulation*/
    real<lower=0> hazard_reg;

    /* number of years */
    int<lower=1> Y; 

    /* number of locations */
    int<lower=1> L;

    /* number of gestation days */
    int<lower=1, upper=366> G;

    /* number of observations */
    int<lower=1> N;                   // max observations per location
    int<lower=0, upper=N> obs[Y, L];  // actual observations per location 

    /* observations */
    int<lower=1, upper=366> day[Y, L, N];
    int<lower=0> redds[Y, L, N];
    vector[N] temp[Y, L];
    int<lower=1> females[Y];
    int<lower=1> fecundity[Y];
}

transformed data { 
    /* total redds by year and location */
    int<lower=0> total_redds[Y, L];

    /* total number of missing temperatures */
    int<lower=1> N_eps = 0;

    /* total number of hazard bins */
    int<lower=1> N_theta;

    for (i in 1:Y) {
	for (j in 1:L) {
	    total_redds[i, j] = sum(redds[i, j, :obs[i, j]]);
	}
    }

    for (i in 1:Y) {
	for (j in 1:L) {
	    int first = day[i, j, 1];
	    int last = day[i, j, obs[i, j]];

	    N_theta += last - first + 1;

	    N_eps += last - first + 1 + G;
	    N_eps -= obs[i, j];
        }
    }
}

parameters {
    simplex[L] theta_loc[Y];
    real mu_temp[Y, L];
    real<lower=0> sigma_temp[Y, L];
    real alpha_egg[Y, L];
    real beta_egg[Y, L];
    vector[N_eps] eps_temp;
    real<upper=0> base_hazard;
    real<lower=0> hazard_slope;
}

transformed parameters {
    real temp_hazard[Y, L];
    vector<lower=0, upper=1>[N_theta] theta_redd;

    /* copy the directly observed values */
    {
        int eps_temp_cursor = 1;
	int theta_redd_cursor = 1;

        for (i in 1:Y) {
	    for (j in 1:L) {
	        int len1 = day[i, j, obs[i, j]] - day[i, j, 1] + 1;
	        int len2 = len1 + G;

	        /* 
	         * this is the underlying Wiener process for  the 
	         * temperature variation during the current year
	         */ 
	        vector[len2] w_temp;

		/*
		 * dayly probability distribution of eggs
		 */
		vector[365] theta = cumulative_sum(rep_vector(1.0, 365)) / 365;

		theta = beta_cdf(
			  theta, 
			  rep_vector(alpha_egg[i, j], size(theta)), 
			  rep_vector(beta_egg[i, j], size(theta))
			);
		for (k in 1:obs[i, j]) {
		    real val = k > 1 ? theta[day[i, j, k]] - theta[day[i, j, k - 1]] : theta[day[i, j, k]];
		    theta_redd[theta_redd_cursor] = val;
		    theta_redd_cursor += 1;
		}
		theta[2:] = theta[2:] - theta[1:364];

	        /*  copy observed data in the right slots */
	        for (k in 1:obs[i, j]) {
	            int t = day[i, j, k] - day[i, j, 1];
	            real tmp;

	            tmp = temp[i, j, k] - temp[i, j, 1];
	            tmp -= mu_temp[i, j] * t; 
	            tmp /= sigma_temp[i, j];

	            w_temp[1 + t] = tmp;
	        }

	        /* fill in between observations */
                for (k in 2:obs[i, j]) {
	            int t0 = day[i, j, k - 1] - day[i, j, 1];
	            int t1 = day[i, j, k] - day[i, j, 1];

	            for (t in (t0 + 1):(t1 - 1)) {
	        	real alpha = 1.0 / (t1 - t + 1);
	        	real beta = 1.0 - alpha;
	        	real tmp;
	        	
	        	tmp = alpha * w_temp[t] + beta * w_temp[1 + t1]; 
	        	tmp += alpha * beta * eps_temp[eps_temp_cursor];
	        	
	        	eps_temp_cursor += 1;
	        	w_temp[1 + t] = tmp;
	            }
	        }

	        /* fill in one gestation period past last observation */
	        {
	            int start = len1;
	            int end = len2;

	            for (t in (start + 1):end) {
	                w_temp[t] = w_temp[t - 1] + eps_temp[eps_temp_cursor];
	                eps_temp_cursor += 1;
	            }
	        }

	        /* compute the daily temperature hazard */
                {
	            vector[len2] hazard = w_temp;
	            real total_hazard;

	            hazard *= sigma_temp[i, j];
	            hazard += mu_temp[i, j] * (cumulative_sum(rep_vector(1.0, len2)) - 1.0);
	            hazard += temp[i, j, 1];
	            hazard = hazard .* hazard_modulation(hazard, temp_threshold, hazard_reg);
	            hazard *= -hazard_slope;
	            hazard += cumulative_sum(hazard);

	            total_hazard = log_sum_exp(
			log(theta[day[i, j, 1]:day[i, j, obs[i, j]]]) + 
			hazard[(G + 1):] - hazard[:(len2 - G)]
		    );
	            total_hazard += base_hazard;

	            temp_hazard[i, j] = total_hazard;
	        }
	    }
        }
    }
}

model {
    /*
     * the temperature variation in a given year is modeled
     * as a brownian motion with drift
     */
    for (i in 1:Y) {
	int start = obs_roi[i, 1];
	int end = obs_roi[i, 2];
	int len = end - start + 1;
	vector[len] delta_temp = temp[(start + 1):end] - temp[start:(end - 1)];
	vector[len] delta_time = day[(start + 1):end] - day[start:(end - 1)];
	vector[len] mu = mu_temp[i] * delta_time;
	vector[len] sigma = sigma_temp[i] * sqrt(delta_time);

        delta_temp ~ normal(mu, sigma);
    }

    eps_temp ~ std_normal();

    /*
     * redd creation model
     */ 
    {
	int old = 1;
        for (i in 1:Y) {
            total_redds[i, :] ~ multinomial(theta_loc[i]);

            for (j in 1:L) {
                int new = old + obs[i, j];
                redds[i, j, :obs[i,j]] ~ multinomial(theta_redds[old:(new - 1)]);
                old = new;
            }
        }
    }
}
