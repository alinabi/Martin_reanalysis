functions {

    vector mean_temp(int year, real mu0, real mu1, real phi) {
	vector[365] res = cumulative_sum(rep_vector(2*pi()/365, 365));
	res += 2 * pi() * (year - 1);
	return mu0 + mu1 * sin(res + phi);
    }

    vector egg_dist(real mu, real sigma) {
	vector[365] res = cumulative_sum(rep_vector(1.0, 365));
	res = (res/365 - mu)/sigma;
	return softmax(-res .* res);
    }

    vector temp_hazard(vector t, real t_crit, real beta, real eta) {
	vector[num_elements(t)] res = log(1 + exp((t - t_crit) / eta));
	return beta * cumulative_sum(res);
    }

    vector maturation_rate(vector t, real alpha, real beta) {
	vector[num_elements(t)] res = alpha + beta * t;
	return cumulative_sum(res);
    }
}


data {
    int<lower=1> Y; // number of years
    int<lower=1> L; // number of locations
    int<lower=1> N; // number of redd and temp observations
    int<lower=1> S; // number of survival observations

    /* redd and temperature observations */
    int<lower=1, upper=Y> obs_year[N];
    int<lower=1, upper=L> obs_location[N];
    int<lower=1, upper=365> obs_day[N];
    int<lower=0> obs_redds[N];
    vector<offset=10>[N] obs_temp;

    /* population observations */
    int<lower=1, upper=Y> surv_year[S];
    int<lower=1> surv_eggs[S];
    int<lower=1> surv_fry[S];

    /* redd and temp data indexing */    
    int<lower=0, upper=N> obs_offset[L, Y];
    int<lower=0, upper=N> obs_length[L, Y];

    /* assumed egg maturation model */
    real<lower=0> beta_egg;
    real alpha_egg;

    /* regularization for the temperature modulation function */
    real<lower=0> temp_reg;
}

transformed data {
    int<lower=1, upper=Y*365> obs_time[N];
    
    for (i in 1:N) {
	obs_time[i] = 365 * (obs_year[i] - 1) + obs_day[i];
    }
}


parameters {
    /* parameters for local temperature profile */
    real<offset=10> mu0_temp;
    real<lower=0> mu1_temp;
    vector[L] temp_bias;
    real<lower=-pi(), upper=pi()> phi_temp;

    /* temperature profile */
    vector<offset=10>[365] temp[Y];

    /* temperature variability */
    real<lower=0> sigma_temp;

    /* hazard model */
    real temp_crit;
    real<lower=0> beta_hazard;

    /* annual_egg_distribution */
    real mu_egg;
    real<lower=0> sigma_egg;

    /* survival regression model */
    real<upper=0> base_survival;

}

transformed parameters {
    /* annual temperature and egg profiles */	
    vector[365] mu_temp[Y] ;
    vector[365] egg_prob;

    /* survival regression */
    vector<upper=0>[Y] survival; 

    egg_prob = egg_dist(mu_egg, sigma_egg);

    for (y in 1:Y) {
        mu_temp[y] = mean_temp(y, mu0_temp, mu1_temp, phi_temp);
    }

    for (y in 1:Y) {
        vector[365] hzd = temp_hazard(temp[y], temp_crit, beta_hazard, temp_reg);
        vector[365] dev = maturation_rate(temp[y], alpha_egg, beta_egg);
        vector[365] hazard = rep_vector(100, 365); 
    
        for (i in 1:365) {
            for (j in i:365) {
                if (dev[j] - dev[i] >= 1.0) {
		    hazard[i] = hzd[j] - hzd[i];
		    break;
		}
            }
        }

	survival[y] = base_survival + log_sum_exp(log(egg_prob) - hazard); 
    }


}


model {
    for (l in 1:L) {
	for (y in 1:Y) {
	    int start = obs_offset[l, y];
	    int stop = obs_offset[l, y] + obs_length[l, y] - 1;
	    vector[obs_length[l, y]] theta;

	    if (start == 0) continue;

	    for (i in start:stop) {
	        int j = i - start + 1;
	        if (j == 1) {
	            theta[j] = sum(egg_prob[:obs_day[i]]);
	        }
	        else {
	            theta[j] = sum(egg_prob[(obs_day[i - 1] + 1):obs_day[i]]);
	        }
	    }
	    theta /= sum(theta);

	    obs_redds[start:stop] ~ multinomial(theta);
	}
    }

    /* temperature model */

    temp_bias ~ normal(0, 1);
    mu1_temp ~ normal(0, 1e-2);

    for (l in 1:L) {
	for (y in 1:Y) {
	    int start = obs_offset[l, y];
	    int stop = start + obs_length[l, y] - 1;
	    int n = obs_length[l, y];
	    vector[n] mu;

	    if (start == 0) continue;
	    mu = temp_bias[l] + mu_temp[y][obs_day[start:stop]];
            obs_temp[start:stop] ~ normal(mu, sigma_temp);
	}
    }

    for (y in 1:Y) {
	temp[y] ~ normal(mu_temp[y], sigma_temp);
    }

    /* survival model */
    surv_fry ~ binomial(surv_eggs, exp(survival[surv_year]));
}
