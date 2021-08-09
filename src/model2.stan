functions {

    vector mean_temp(int n, real mu0, real mu1, real mu2, real phi) {
	vector[n] res = cumulative_sum(rep_vector(1.0, n));
	res *= 2 * pi() / 365.0;
	return mu0 + mu1 * res + mu2 * sin(res + phi);
    }

    vector egg_dist(real mu, real sigma) {
	vector[365] res = cumulative_sum(rep_vector(1.0, 365));
	res /= 365.0;
	res = Phi_approx((logit(res) - mu) / sigma);
	res /= res[365];
	res[2:365] -= res[1:364];  
	return res;
    }

    vector temp_hazard(vector t, real t_crit, real beta, real eta) {
	return beta * log(1 + exp((t - t_crit) / eta));
    }
}


data {
    int<lower=1> Y; // number of years
    int<lower=1> L; // number of locations
    int<lower=1> G; // number of gestation days
    int<lower=1> N; // number of redd and temp observations
    int<lower=1> P; // number of survival observations
    int<lower=Y> S; // number of series

    /* redd and temperature observations */
    int<lower=1, upper=Y> obs_year[N];
    int<lower=1, upper=L> obs_location[N];
    int<lower=1, upper=365> obs_day[N];
    int<lower=0> obs_redds[N];
    vector[N] obs_temp;

    /* population observations */
    int<lower=1, upper=Y> surv_year[P];
    int<lower=1> surv_eggs[P];
    int<lower=1> surv_parr[P];

    /* redd and temp data indexing */    
    int<lower=1, upper=N> obs_series_span[S, 2];

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
    vector<offset=10>[L] mu0_temp;
    vector[L] mu1_temp;
    vector<lower=0>[L] mu2_temp;
    real<lower=-pi(), upper=pi()> phi_temp;

    /* temperature variability */
    real<lower=0> sigma_temp;

    /* temperature imputation */
    vector<offset=10>[(365*Y+G)*L - N] imp_temp;

    /* hazard model */
    real<offset=10> temp_crit;
    real<lower=0> beta_hazard;

    /* annual_egg_distribution */
    //real mu_egg;
    //real<lower=0> sigma_egg;

    /* survival regression model */
    real<upper=0> alpha_survival;
    vector<lower=0>[L] beta_survival;

    simplex[365] egg_prob;
}

transformed parameters {
    /* annual temperature and egg profiles */	
    vector[365*Y + G] mu_temp[L] ;
    //simplex[365] egg_prob;

    /* imputed temperature data */
    vector[365*Y + G] temp[L];

    /* temperature hazard */
    vector[365*Y + G] hazard[L];

    /* survival regression */
    matrix<upper=0>[Y, L] loc_survival;
    vector<upper=0>[L] survival; 


    //egg_prob = egg_dist(mu_egg, sigma_egg);

    for (l in 1:L) {
	int n = 365 * Y + G;
        mu_temp[l] = mean_temp(n, mu0_temp[l], mu1_temp[l], mu2_temp[l], phi_temp);
    }

    {
	int imp_curs = 1;
	int obs_curs = 1;
	for (l in 1:L) {
	    for (t in 1:(365*Y+G)) {
	        if (obs_curs <= N && t == obs_time[obs_curs]) {
	    	    temp[l][t] = obs_temp[obs_curs];
	    	    obs_curs += 1;
	        }
	        else {
	    	    temp[l][t] = imp_temp[imp_curs];
	    	    imp_curs += 1;
	        }
	    }
	}
    }

    for (l in 1:L) {
	hazard[l] = temp_hazard(temp[l], temp_crit, beta_hazard, temp_reg);
    }

    for (l in 1:L) {
	for (y in 1:Y) {
	    int start = 365 * (y - 1) + 1;
	    int stop = 365 * y;
	    vector[365+G] hzd;

	    hzd = hazard[l][start:(stop+G)];
	    hzd = cumulative_sum(hzd);
	    hzd[1:365] = hzd[(1+G):(365+G)] - hzd[1:365];

	    loc_survival[y, l] = log_sum_exp(log(egg_prob) - hzd[1:365]);
	}
    }

    survival = alpha_survival + loc_survival * beta_survival;

}


model {
    for (s in 1:S) {
	int start = obs_series_span[s, 1];
	int end = obs_series_span[s, 2];
	int n = end - start + 1;
	vector[n] theta;

	for (i in start:end) {
	    int j = i - start + 1;
	    if (i == start) {
	        theta[j] = sum(egg_prob[:obs_day[i]]);
	    }
	    else {
	        theta[j] = sum(egg_prob[(obs_day[i - 1] + 1):obs_day[i]]);
	    }
	}
	theta /= sum(theta);

	obs_redds[start:end] ~ multinomial(theta);
    }

    for (l in 1:L) {
	int n = 365 * Y + G - 1;
        vector[n] mu = mu_temp[l][2:] - mu_temp[l][:n];
        temp[l][2:] - temp[l][:n] ~ normal(mu, sigma_temp);
    }

    {
	int y[P] = surv_year;
	surv_parr ~ binomial(surv_eggs, exp(survival[y]));
    }
}
