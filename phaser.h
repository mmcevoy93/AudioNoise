struct {
	struct lfo_state lfo;
	struct biquad_coeff coeff;
	struct biquad_state ph1, ph2, ph3;
	float center_f, octaves, Q;
	float feedback, prev;
} phaser;

#define linear(pot, a, b) ((a)+pot*((b)-(a)))
#define cubic(pot, a, b) linear((pot)*(pot)*(pot), a, b)

void phaser_init(float pot1, float pot2, float pot3, float pot4)
{
	float ms = cubic(pot1, 25, 2000);		// 25ms .. 2s
	set_lfo_ms(&phaser.lfo, ms);
	phaser.feedback = linear(pot2, 0, 0.75);

	pot3 = 2*pot3;
	phaser.center_f = linear(pot3*pot3*pot3, 50, 880);	// 50Hz .. 1kHz
	phaser.octaves = 2;
	phaser.Q = linear(pot4, 0.25, 2);

	fprintf(stderr, "phaser:");
	fprintf(stderr, " lfo=%g ms", ms);
	fprintf(stderr, " center_f=%g Hz", phaser.center_f);
	fprintf(stderr, " feedback=%g", phaser.feedback);
	fprintf(stderr, " Q=%g\n", phaser.Q);
}

float phaser_step(float in)
{
	float lfo = lfo_step(&phaser.lfo, lfo_triangle);
	float freq = fastpow(2, lfo*phaser.octaves) * phaser.center_f;
	float out;

	_biquad_allpass_filter(&phaser.coeff, freq, phaser.Q);

	out = _biquad_step(&phaser.coeff, &phaser.ph1, in + phaser.prev * phaser.feedback);
	out = _biquad_step(&phaser.coeff, &phaser.ph2, out);
	out = _biquad_step(&phaser.coeff, &phaser.ph3, out);
	phaser.prev = out;

	return limit_value(in + out);
}
