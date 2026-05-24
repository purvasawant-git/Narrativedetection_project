document.addEventListener('DOMContentLoaded', () => {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const views = document.querySelectorAll('.dash-view');
    const localFilters = document.querySelectorAll('.dash-filter');

    // Global state
    window.currentView = 'dash-quality';

    // Handle Tab Navigation
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            views.forEach(v => v.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');

            window.currentView = targetId;
            
            // Re-render using the specific local filter for this tab
            const localFilterEl = document.getElementById(`filter-${targetId.split('-')[1]}`);
            const filterValue = localFilterEl ? localFilterEl.value : 'all';
            
            if (window.renderChartForView) {
                window.renderChartForView(targetId, filterValue);
            }
        });
    });

    // Handle Localized Duration Filters
    localFilters.forEach(filter => {
        filter.addEventListener('change', (e) => {
            const viewId = e.target.getAttribute('data-view');
            const filterValue = e.target.value;
            
            // Only re-render if it's the currently active view (or just force re-render anyway)
            if (window.renderChartForView && window.currentView === viewId) {
                window.renderChartForView(viewId, filterValue);
            }
        });
    });
});
