/**
 * home.js - Home page functionality
 */

// DataTable instance
let attendanceTable;

// Chart instances
let dailyAttendanceChart;
let studentSuccessChart;

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Initialize DataTable
    initializeAttendanceTable();
    
    // Load summary data
    loadSummaryData();
    
    // Load daily attendance chart
    loadDailyAttendanceChart();
    
    // Load student success rate chart
    loadStudentSuccessRateChart();
    
    // Set up event listeners
    setupEventListeners();
});

/**
 * Initialize the attendance DataTable
 */
function initializeAttendanceTable() {
    attendanceTable = $('#attendanceTable').DataTable({
        ajax: {
            url: '/api/attendance-data',
            dataSrc: function(json) {
                return json.data || [];
            }
        },
        columns: [
            { data: 'student_id' },
            { data: 'student_name' },
            { 
                data: 'timestamp',
                render: function(data) {
                    const date = new Date(data);
                    return date.toLocaleString('vi-VN');
                }
            },
            { 
                data: 'status',
                render: function(data) {
                    if (data == 1) {
                        return '<span class="status-badge status-success">Thành công</span>';
                    } else {
                        return '<span class="status-badge status-failure">Thất bại</span>';
                    }
                }
            }
        ],
        order: [[2, 'desc']], // Sort by timestamp desc
        pageLength: 10,
        language: {
            url: '//cdn.datatables.net/plug-ins/1.13.6/i18n/vi.json'
        },
        responsive: true
    });
}

/**
 * Load summary data for the dashboard
 */
function loadSummaryData() {
    fetch('/api/attendance-summary')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update summary cards
                document.getElementById('uniqueStudents').textContent = data.data.unique_students;
                document.getElementById('successfulRecognitions').textContent = data.data.successful_recognitions;
                document.getElementById('failedRecognitions').textContent = data.data.failed_recognitions;
                document.getElementById('successRate').textContent = data.data.success_rate + '%';
            }
        })
        .catch(error => {
            console.error('Error loading summary data:', error);
            showErrorAlert('Không thể tải dữ liệu tổng quan');
        });
}

/**
 * Load daily attendance chart
 */
function loadDailyAttendanceChart() {
    fetch('/api/attendance-by-date')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                const chartData = data.data;
                
                // Prepare chart data
                const labels = chartData.map(item => item.date);
                const successful = chartData.map(item => item.successful);
                const failed = chartData.map(item => item.failed);
                
                // Create datasets
                const datasets = [
                    {
                        label: 'Thành công',
                        backgroundColor: chartColors.successLight,
                        borderColor: chartColors.success,
                        borderWidth: 2,
                        data: successful
                    },
                    {
                        label: 'Thất bại',
                        backgroundColor: chartColors.dangerLight,
                        borderColor: chartColors.danger,
                        borderWidth: 2,
                        data: failed
                    }
                ];
                
                // Create chart
                dailyAttendanceChart = createBarChart('dailyAttendanceChart', labels, datasets, {
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Ngày'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Số lượng'
                            }
                        }
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error loading daily attendance chart:', error);
            showErrorAlert('Không thể tải biểu đồ điểm danh theo ngày');
        });
}

/**
 * Load student success rate chart
 */
function loadStudentSuccessRateChart() {
    fetch('/api/student-success-rate')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                const chartData = data.data;
                
                // Prepare chart data
                const labels = chartData.map(item => item.student_name);
                const successRates = chartData.map(item => item.success_rate);
                
                // Generate gradient colors based on success rate
                const backgroundColors = successRates.map(rate => {
                    if (rate >= 80) return chartColors.successLight;
                    if (rate >= 50) return chartColors.warningLight || 'rgba(243, 156, 18, 0.2)';
                    return chartColors.dangerLight;
                });
                
                const borderColors = successRates.map(rate => {
                    if (rate >= 80) return chartColors.success;
                    if (rate >= 50) return chartColors.warning || '#f39c12';
                    return chartColors.danger;
                });
                
                // Create datasets
                const datasets = [
                    {
                        label: 'Tỷ lệ thành công (%)',
                        backgroundColor: backgroundColors,
                        borderColor: borderColors,
                        borderWidth: 2,
                        data: successRates
                    }
                ];
                
                // Create chart
                studentSuccessChart = createBarChart('studentSuccessChart', labels, datasets, {
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Sinh viên'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Tỷ lệ (%)'
                            }
                        }
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error loading student success rate chart:', error);
            showErrorAlert('Không thể tải biểu đồ tỷ lệ thành công theo sinh viên');
        });
}

/**
 * Set up event listeners for filters and buttons
 */
function setupEventListeners() {
    // Student filter
    document.getElementById('studentFilter').addEventListener('input', function() {
        attendanceTable.search(this.value).draw();
    });
    
    // Date filter
    document.getElementById('dateFilter').addEventListener('change', function() {
        const filterValue = this.value;
        
        // Custom filtering function
        $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {
            if (!filterValue) {
                return true; // No filter
            }
            
            // Get timestamp from the date column (index 2)
            const timestamp = data[2];
            const recordDate = new Date(timestamp).toISOString().split('T')[0];
            
            return recordDate === filterValue;
        });
        
        // Apply filter
        attendanceTable.draw();
        
        // Remove custom filtering function
        $.fn.dataTable.ext.search.pop();
    });
    
    // Reset filters button
    document.getElementById('resetFilters').addEventListener('click', function() {
        document.getElementById('studentFilter').value = '';
        document.getElementById('dateFilter').value = '';
        attendanceTable.search('').draw();
    });
    
    // Refresh data button
    document.getElementById('refreshBtn').addEventListener('click', function() {
        // Show loading overlay
        document.getElementById('loadingOverlay').classList.remove('d-none');
        
        // Call refresh API
        fetch('/api/refresh', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            document.getElementById('loadingOverlay').classList.add('d-none');
            
            if (data.status === 'success') {
                // Show success message
                Swal.fire({
                    icon: 'success',
                    title: 'Thành công!',
                    text: 'Dữ liệu đã được cập nhật từ WandB.',
                    timer: 2000,
                    showConfirmButton: false
                });
                
                // Refresh page data
                attendanceTable.ajax.reload();
                loadSummaryData();
                loadDailyAttendanceChart();
                loadStudentSuccessRateChart();
            } else {
                // Show error message
                showErrorAlert(data.message || 'Không thể cập nhật dữ liệu');
            }
        })
        .catch(error => {
            // Hide loading overlay
            document.getElementById('loadingOverlay').classList.add('d-none');
            
            console.error('Error refreshing data:', error);
            showErrorAlert('Không thể kết nối với API để cập nhật dữ liệu');
        });
    });
}

/**
 * Show error alert
 * @param {string} message - Error message to display
 */
function showErrorAlert(message) {
    Swal.fire({
        icon: 'error',
        title: 'Lỗi!',
        text: message,
        confirmButtonText: 'Đóng'
    });
} 
