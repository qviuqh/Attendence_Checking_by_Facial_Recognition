/**
 * charts.js - Common chart utilities and configurations
 */

// Chart color scheme
const chartColors = {
    primary: '#3498db',
    secondary: '#2c3e50',
    success: '#2ecc71',
    danger: '#e74c3c',
    warning: '#f39c12',
    info: '#1abc9c',
    light: '#ecf0f1',
    dark: '#34495e',
    primaryLight: 'rgba(52, 152, 219, 0.2)',
    dangerLight: 'rgba(231, 76, 60, 0.2)',
    successLight: 'rgba(46, 204, 113, 0.2)',
};

// Common chart options
const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 20,
                font: {
                    size: 12
                }
            }
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            titleFont: {
                size: 14
            },
            bodyFont: {
                size: 13
            },
            padding: 10,
            cornerRadius: 5,
            displayColors: true
        }
    }
};

/**
 * Create or update a bar chart
 * @param {string} canvasId - The canvas element ID
 * @param {Array} labels - Chart labels (X-axis)
 * @param {Array} datasets - Chart datasets
 * @param {Object} options - Additional chart options
 * @returns {Chart} - Chart instance
 */
function createBarChart(canvasId, labels, datasets, options = {}) {
    const chartElement = document.getElementById(canvasId);
    
    if (!chartElement) {
        console.error(`Canvas element with ID '${canvasId}' not found.`);
        return null;
    }
    
    // Clear existing chart if any
    if (chartElement.chart) {
        chartElement.chart.destroy();
    }
    
    // Create chart
    const ctx = chartElement.getContext('2d');
    const chartOptions = {...commonChartOptions, ...options};
    
    chartElement.chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: chartOptions
    });
    
    return chartElement.chart;
}

/**
 * Create or update a line chart
 * @param {string} canvasId - The canvas element ID
 * @param {Array} labels - Chart labels (X-axis)
 * @param {Array} datasets - Chart datasets
 * @param {Object} options - Additional chart options
 * @returns {Chart} - Chart instance
 */
function createLineChart(canvasId, labels, datasets, options = {}) {
    const chartElement = document.getElementById(canvasId);
    
    if (!chartElement) {
        console.error(`Canvas element with ID '${canvasId}' not found.`);
        return null;
    }
    
    // Clear existing chart if any
    if (chartElement.chart) {
        chartElement.chart.destroy();
    }
    
    // Create chart
    const ctx = chartElement.getContext('2d');
    const chartOptions = {...commonChartOptions, ...options};
    
    chartElement.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: chartOptions
    });
    
    return chartElement.chart;
}

/**
 * Create or update a pie/doughnut chart
 * @param {string} canvasId - The canvas element ID
 * @param {Array} labels - Chart labels
 * @param {Array} data - Chart data values
 * @param {Array} backgroundColor - Background colors for each segment
 * @param {boolean} isDoughnut - Whether to create a doughnut chart
 * @param {Object} options - Additional chart options
 * @returns {Chart} - Chart instance
 */
function createPieChart(canvasId, labels, data, backgroundColor, isDoughnut = false, options = {}) {
    const chartElement = document.getElementById(canvasId);
    
    if (!chartElement) {
        console.error(`Canvas element with ID '${canvasId}' not found.`);
        return null;
    }
    
    // Clear existing chart if any
    if (chartElement.chart) {
        chartElement.chart.destroy();
    }
    
    // Create chart
    const ctx = chartElement.getContext('2d');
    const chartOptions = {...commonChartOptions, ...options};
    
    chartElement.chart = new Chart(ctx, {
        type: isDoughnut ? 'doughnut' : 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColor,
                borderWidth: 1
            }]
        },
        options: chartOptions
    });
    
    return chartElement.chart;
}

/**
 * Format date string to a more readable format
 * @param {string} dateString - Date string in ISO format
 * @returns {string} - Formatted date string
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('vi-VN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
    });
}

/**
 * Format time string to a more readable format
 * @param {string} timeString - Time string from date
 * @returns {string} - Formatted time string
 */
function formatTime(timeString) {
    const date = new Date(timeString);
    return date.toLocaleTimeString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit'
    });
}