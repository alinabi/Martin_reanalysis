data {
    int<lower=1> Y;
    int<lower=1> L;
    int<lower=1> G;

    int<lower=1> females[Y];
    int<lower=1> fecundity[Y];
    int<lower=1> parr[2, Y];

    int<lower=1> N;
    int<lower=1, upper=Y> obs_year[N];
    int<lower=1, upper=L> obs_loc[N];
    int<lower=1, upper=365> obs_day[N];
    int<lower=0> obs_redds[N];
    vector[N] obs_temp;

}


transformed data {
    real<lower=0> temp_reg = 0.001;

    int<lower=1> year_len = 365;
    vector[year_len] year_phi;

    int<lower=1> obs_roi[Y, L, 2];
    int<lower=1> missing_roi[Y, L, 2];
    int<lower=1, upper=365> missing_days[Y*L*365 - N];

    int<lower=0> loc_redds[Y, L];

    int<lower=1> eggs[Y];

    year_phi = cumulative_sum(rep_vector(1.0, year_len));
    year_phi = 2 * pi() * year_phi / year_len;

    {
	int last_year = -1;
	int last_loc = -1;
	for (i in 1:N) {
	    if (obs_year[i] != last_year) {

		obs_roi[obs_year[i], obs_loc[i], 1] = i;
	    }
	    obs_roi[obs_year[i], obs_loc[i], 2] = i;
	}
    }

    {
	int m_curs = 1;

        for (y in 1:Y) {
	    for (l in 1:L) {
		int start = obs_roi[y, l, 1];
		int end = obs_roi[y, l, 2];
		int o_curs = start;

		missing_roi[y, l, 1] = m_curs;

	        for (d in 1:year_len) {

	            if (o_curs > end) {
	                missing_days[m_curs] = d;
			missing_roi[y, l, 2] = m_curs;
	                m_curs += 1;
	                continue;
	            }

                    if (d == obs_day[o_curs]) {
                        o_curs += 1;
			continue;
		    }

		    missing_days[m_curs] = d;
		    missing_roi[y, l, 2] = m_curs;
	            m_curs += 1;
	        }
	    }
        }
    }

    for (y in 1:Y) {
	for (l in 1:L) {
	    int start = obs_roi[y, l, 1];
	    int end = obs_roi[y, l, 2];
	    loc_redds[y, l] = sum(obs_redds[start:end]);
	}
    }

    for (y in 1:Y) {
	eggs[y] = females[y] * fecundity[y];
    }
}


parameters {
    /* mean temperature profile */
    real mu0_temp[L];
    real mu1_temp[L];
    real<lower=-pi(), upper=pi()> phi_temp[L];
    real<lower=0> sigma_temp[L];
    vector[Y*L*year_len - N] temp_zeta;

    /* temperature related hazard parameters */
    real temp_threshold;
    real<lower=0> base_hazard;
    real<lower=0> hazard_slope;

    /* daily egg distribution parameters */
    real egg_alpha;
    real egg_beta;
    simplex[L] egg_loc_dist;
}


transformed parameters {
    /* full temperature data */
    vector[year_len] mu_temp[L];	
    vector[year_len] temp[Y, L];

    /* full hazard data */
    vector<lower=0>[year_len] hazard[Y, L];
    vector<lower=0>[L] annual_hazard[Y];
    vector<lower=0>[Y] total_hazard;

    /* discretized daily egg distribution */
    vector<lower=0, upper=1>[year_len] egg_cdf;

    /* temperature calculation */
    {
        int curs = 1;
        for (y in 1:Y) {
            for (l in 1:L) {
                int start = obs_roi[y, l, 1];
                int end = obs_roi[y, l, 2];
                int m_start = missing_roi[y, l, 1];
		int m_end = missing_roi[y, l, 2];
		int m_span = m_end - m_start + 1;

                temp[y, l][obs_day[start:end]] = obs_temp[start:end];
                temp[y, l][missing_days[m_start:m_end]] = temp_zeta[curs:(curs + m_span - 1)];
                curs += m_span;

		mu_temp[l]  = rep_vector(mu0_temp[l], year_len);
		mu_temp[l] += mu1_temp[l] * sin(year_phi + phi_temp[l]);
            }
        }
    }  

    

    /* daily hazard calculation */
    for (y in 1:Y) {
        for (l in 1:L) {
	    hazard[y, l]  = 0.5 * temp[y, l] .* (1 + tanh((temp[y, l] - temp_threshold) / temp_reg));
	    hazard[y, l] *= -hazard_slope;
	}
    }

    /* daily egg distribution calculation */
    {
        real len = year_len;
        for (d in 1:(year_len - 1)) {
            egg_cdf[d] = beta_cdf(d / len, egg_alpha, egg_beta);
        }
	egg_cdf[year_len] = 1;
    }

    /* total and annual hazard calculation */
    {
        vector[year_len] theta;
	theta[1] = egg_cdf[1];
	theta[2:] = egg_cdf[2:] - egg_cdf[:(year_len - 1)];

	for (y in 1:Y) {
	    for (l in 1:L) {
	        vector[year_len + G] hzd;

		hzd[:year_len] = hazard[y, l];
		hzd[(year_len + 1):] = hazard[y, l][:G];
		hzd = cumulative_sum(hzd);

	        annual_hazard[y][l] = base_hazard;
		annual_hazard[y][l] += log_sum_exp(theta + hzd[G:] - hzd[:year_len]);
	    }

	    total_hazard[y] = log_sum_exp(log(egg_loc_dist[y]) + annual_hazard[y]);
	}
    }
}


model {
    /* Browninan component of temperature process */	
    for (y in 1:Y) {
	for (l in 1:L) {
	    vector[year_len - 1] d_temp;
	    vector[year_len - 1] mu;

	    d_temp = temp[y, l][2:] - temp[y, l][:(year_len - 1)];
	    mu = mu_temp[l][2:] - mu_temp[l][:(year_len - 1)]; 
	    
	    target += normal_lpdf(d_temp | mu, sigma_temp[l]);
	}
    }

    /* egg distribution */
    for (y in 1:Y) {
	
        loc_redds[y, :] ~ multinomial(egg_loc_dist);

	for (l in 1:L) {
	    int start = obs_roi[y, l, 1];
	    int end = obs_roi[y, l, 2];

	    obs_redds[start:end] ~ multinomial(egg_cdf);

	}
    }

    /* survival observation probability model */
    {
	vector[Y] theta = exp(total_hazard);
	vector[Y] high;
	vector[Y] low;

	for (y in 1:Y) {
            high[y] = binomial_cdf(parr[2, y], eggs[y], theta[y]);
	    low[y] = binomial_cdf(parr[1, y], eggs[y], theta[y]);
	}

	target += sum(log_diff_exp(high, low));
    }
}
