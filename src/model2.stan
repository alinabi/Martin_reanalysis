functions {

    vector mean_temp(real mu0, real mu1, real phi) {
	vector[365] res = cumulative_sum(rep_vector(1.0, 365));
	res *= 2 * pi() / 365.0;
	return mu0 + mu1 * sin(res + phi);
    }

    vector egg_dist(real mu, real sigma) {
	vector[365] res = cumulative_sum(rep_vector(1.0, 365));
	res /= 365.0;
	res = Phi_approx((logit(res) - mu) / sigma);
	res /= res[365];
	res[2:365] -= res[1:364];  
	return res;
    }

    vector temp_hazard(vector t, real t_crit, real eta) {
	vector[size(t)] res = log(1 + exp((t - t_crit) / eta));
	return res;
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
    int<lower=1, upper=N> obs_year_span[Y, 2];

    /* regularization for the temperature modulation function */
    real<lower=0> temp_reg;
}


parameters {
    /* parameters for local temperature profile */
    vector<offset=10>[L] mu0_temp;
    vector<lower=0>[L] mu1_temp;
    real<lower=-pi(), upper=pi()> phi_temp;

    /* temperature variability */
    real<lower=0> sigma_temp;

    /* temperature imputation */
    vector<offset=10>[365*Y*L - N] imp_temp;

    /* hazard model */
    real<offset=10> temp_crit;
    real<lower=0> beta_hazard;

    /* annual_egg_distribution */
    real mu_egg;
    real<lower=0> sigma_egg;

    /* survival regression model */
    real<upper=0> alpha_survival;
    vector<lower=0>[L] beta_survival;

}

transformed parameters {
    /* annual temperature and egg profiles */	
    vector[365] mu_temp[L];
    simplex[365] egg_prob;

    /* imputed temperature data */
    vector[365] temp[Y, L];

    /* survival regression */
    matrix<upper=0>[Y, L] loc_survival;
    vector<upper=0>[L] survival; 


    egg_prob = egg_dist(mu_egg, sigma_egg);

    for (l in 1:L) {
        mu_temp[l] = mean_temp(mu0_temp[l], mu1_temp[l], phi_temp);
    }

    {
	int imp_curs = 1;
	int obs_curs = 1;
	for (y in 1:Y) {
	    for(l in 1:L) {
		for (d in 1:365) {
		    if (obs_curs <= N && y == obs_year[obs_curs] && l == obs_location[obs_curs] && d == obs_day[obs_curs]) {
			temp[y, l][d] = obs_temp[obs_curs];
			obs_curs += 1;
		    }
		    else {
			temp[y, l][d] = imp_temp[imp_curs];
			imp_curs += 1;
		    }
		}
	    }
	}
    }

    for (y in 1:Y) {
	for (l in 1:L) {
	    vector[365] hzd;
	    vector[365-G] h;
	    vector[365-G] p;

	    hzd = temp_hazard(temp[y, l], temp_crit, temp_reg);
	    hzd = cumulative_sum(hzd);
	    h = beta_hazard * (hzd[(G+1):365] - hzd[1:(365-G)]);  

	    p = egg_prob[1:(365-G)] / sum(egg_prob[1:(365-G)]);

	    loc_survival[y, l] = log_sum_exp(log(p) - h);
	}
    }

    survival = alpha_survival + loc_survival * beta_survival;

}


model {
    print("checkpoint 1: ", target());


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
    print("checkpoint 2: ", target());

    for (y in 1:Y) {
	for (l in 1:L) {
	  vector[364] mu = mu_temp[l][2:365] - mu_temp[l][1:364];
          temp[y, l][2:365] - temp[y, l][1:364] ~ normal(mu, sigma_temp);
        }
    }
    print("checkpoint 3: ", target());

    {
	int y[P] = surv_year;
	surv_parr ~ binomial(surv_eggs, exp(survival[y]));
    }
    print("checkpoint 4: ", target());
}
