functions {

    vector egg_dist(int n, real mu, real sigma) {
	vector[n] res = cumulative_sum(rep_vector(1.0, n));
	res = (res/n - mu)/sigma;
	return softmax(-res .* res);
    }

    vector temp_hazard(vector t, real t_crit, real beta, real eta) {
	vector[num_elements(t)] res = log(1 + exp((t - t_crit) / eta));
	return beta * res;
    }

    vector maturation_rate(vector t, real alpha, real beta) {
	vector[num_elements(t)] res = alpha + beta * t;
	return cumulative_sum(res);
    }
}


data {
    int<lower=1> Y; // number of years
    int<lower=1> R; // number of redd observation
    int<lower=1> D; // number of days in incubation period
    int<lower=1> T; // number of temp observations
    int<lower=1> S; // number of survival observations

    /* redd observations */
    int<lower=1, upper=Y> redd_year[R];
    int<lower=1, upper=D> redd_day[R];
    int<lower=0> obs_redds[R];

    /* temp observations */
    int<lower=1, upper=Y> temp_year[T];
    vector[T] obs_temp;

    /* population observations */
    int<lower=1, upper=Y> surv_year[S];
    int<lower=1> surv_eggs[S];
    int<lower=1> surv_fry[S];

    /* redd data indexing */    
    int<lower=0, upper=R> redd_offset[Y];
    int<lower=0, upper=R> redd_length[Y];


    /* regularization for the temperature modulation function */
    real<lower=0> temp_reg;

    /* egg maturation regularization */
    real<lower=0> mat_reg;
}

transformed data {
    real temp_off = mean(obs_temp);
    real<lower=0> temp_mult = sd(obs_temp);
}


parameters {
    /* parameters for local temperature profile */
    real<offset=temp_off, multiplier=temp_mult> mu_temp[Y];
    real<lower=0> sigma_temp[Y];

    /* temperature profile */
    vector[D] surv_temp[Y];

    /* temperature variability */

    /* hazard model */
    real temp_crit;
    real<lower=0> beta_hazard;

    /* annual_egg_distribution */
    real<offset=0.5, multiplier=0.5> mu_egg;
    real<lower=0> sigma_egg;

    /* survival regression model */
    real<upper=0> base_survival;

    /* egg maturation model */
    real<lower=0> beta_egg;
    real alpha_egg;

}

transformed parameters {
    /* annual temperature and egg profiles */	
    //vector[365] mu_temp ;
    vector[D] egg_prob;

    /* survival regression */
    vector<upper=0>[Y] survival; 

    egg_prob = egg_dist(D, mu_egg, sigma_egg);

    for (y in 1:Y) {
        vector[D] hzd = temp_hazard(surv_temp[y], temp_crit, beta_hazard, temp_reg);
        vector[D] dev = maturation_rate(surv_temp[y], alpha_egg, beta_egg);
        vector[D] hazard; 
    
        for (i in 1:D) {
	    vector[D - i + 1] mat = (1 - dev[i:D] + dev[i]) / mat_reg;

	    hazard[i] = dot_product(hzd[i:D], inv_logit(mat));
        }

	survival[y] = base_survival + log_sum_exp(log(egg_prob) - hazard); 
    }

}


model {
    for (y in 1:Y) {
	int start = redd_offset[y];
	int stop = redd_offset[y] + redd_length[y] - 1;
	vector[redd_length[y]] theta;

	if (start == 0) continue;

	for (i in start:stop) {
	    int j = i - start + 1;
	    if (j == 1) {
	        theta[j] = sum(egg_prob[:redd_day[i]]);
	    }
	    else {
	        theta[j] = sum(egg_prob[(redd_day[i - 1] + 1):redd_day[i]]);
	    }
	}
	theta /= sum(theta);

	obs_redds[start:stop] ~ multinomial(theta);
    }

    /* temperature model */
    obs_temp ~ normal(mu_temp[temp_year], sigma_temp[temp_year]);

    /* survival model */
    for (y in 1:Y) {
        surv_temp[y] ~ normal(mu_temp[y], sigma_temp[y]);
    }
    surv_fry ~ binomial(surv_eggs, exp(survival[surv_year]));
}
