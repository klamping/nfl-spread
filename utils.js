function smoothedAccuracyWithTotal(n, accuracy, totalPredictions, prior = 0.45, scalingFactor = 0.1) {
    /**
     * Calculate the smoothed (weighted) accuracy with a dynamic smoothing factor.
     *
     * @param {number} n - Number of predictions for the state.
     * @param {number} accuracy - Observed accuracy for the state (between 0 and 1).
     * @param {number} totalPredictions - Total number of predictions across all states.
     * @param {number} prior - The prior accuracy to fall back on for low sample sizes (default is 0.5).
     * @param {number} scalingFactor - Multiplier for determining smoothing based on total predictions (default is 0.1).
     * @returns {number} Smoothed accuracy value.
     */
    const smoothing = totalPredictions * scalingFactor;
    return (smoothing * prior + n * accuracy) / (smoothing + n);
}

function smoothedAccuracy(n, accuracy, prior = 0.45, smoothing = 10) {
    /**
     * Calculate the smoothed (weighted) accuracy.
     *
     * @param {number} n - Number of predictions for the state.
     * @param {number} accuracy - Observed accuracy for the state (between 0 and 1).
     * @param {number} prior - The prior accuracy to fall back on for low sample sizes (default is 0.5).
     * @param {number} smoothing - The smoothing constant to control weight given to small sample sizes.
     * @returns {number} Smoothed accuracy value.
     */
    return (smoothing * prior + n * accuracy) / (smoothing + n);
}

module.exports = {
    smoothedAccuracyWithTotal,
    smoothedAccuracy
};