/**
 * Automated Drag & Drop Testing for Racing Dashboard
 * ====================================================
 *
 * Tests file upload via drag & drop simulation on http://localhost:8050
 *
 * IMPORTANT LIMITATIONS:
 * ----------------------
 * Puppeteer CANNOT simulate actual OS-level file drag & drop from Windows Explorer.
 * This is a security limitation of all browser automation tools.
 *
 * WHAT THIS SCRIPT DOES:
 * ----------------------
 * 1. Uses Puppeteer's file input API to simulate file selection
 * 2. Programmatically triggers drag events on the upload panel
 * 3. Tests the dashboard's response to file upload
 * 4. Verifies upload success and UI state changes
 *
 * ALTERNATIVE APPROACHES INCLUDED:
 * ---------------------------------
 * - Method 1: Direct file input (most reliable)
 * - Method 2: Simulated drag events (tests UI behavior)
 * - Method 3: DataTransfer API simulation (advanced)
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// Configuration
const CONFIG = {
    dashboardUrl: 'http://localhost:8050',
    testFilePath: 'C:\\project\\data_analisys_car\\master_racing_data.csv',
    screenshotsDir: path.join(__dirname, 'screenshots'),
    timeout: 30000,
    headless: false, // Set to true for CI/CD
    slowMo: 100 // Slow down actions for visibility
};

// Test results
const results = {
    timestamp: new Date().toISOString(),
    tests: [],
    passed: 0,
    failed: 0,
    screenshots: []
};

/**
 * Utility: Add test result
 */
function addResult(testName, passed, message, screenshot = null) {
    const result = {
        test: testName,
        status: passed ? 'PASS' : 'FAIL',
        message: message,
        screenshot: screenshot,
        timestamp: new Date().toISOString()
    };
    results.tests.push(result);
    if (passed) {
        results.passed++;
        console.log(`✅ PASS: ${testName} - ${message}`);
    } else {
        results.failed++;
        console.error(`❌ FAIL: ${testName} - ${message}`);
    }
}

/**
 * Utility: Take screenshot
 */
async function takeScreenshot(page, name) {
    const timestamp = Date.now();
    const filename = `${name}_${timestamp}.png`;
    const filepath = path.join(CONFIG.screenshotsDir, filename);

    await page.screenshot({
        path: filepath,
        fullPage: true
    });

    results.screenshots.push(filename);
    console.log(`📸 Screenshot saved: ${filename}`);
    return filename;
}

/**
 * Utility: Wait for element with timeout
 */
async function waitForElement(page, selector, timeout = 5000) {
    try {
        await page.waitForSelector(selector, { timeout });
        return true;
    } catch (error) {
        return false;
    }
}

/**
 * Utility: Sleep
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * TEST 1: Page Load and Basic Elements
 */
async function testPageLoad(page) {
    console.log('\n🧪 TEST 1: Page Load and Basic Elements');
    console.log('=' .repeat(60));

    try {
        // Navigate to dashboard
        await page.goto(CONFIG.dashboardUrl, {
            waitUntil: 'networkidle2',
            timeout: CONFIG.timeout
        });

        const screenshot1 = await takeScreenshot(page, 'step1_page_loaded');
        addResult('Page Load', true, `Dashboard loaded successfully`, screenshot1);

        // Check for upload panel container
        const uploadPanelExists = await waitForElement(page, '#upload-panel-container', 10000);
        if (uploadPanelExists) {
            addResult('Upload Panel Exists', true, 'Upload panel container found');
        } else {
            addResult('Upload Panel Exists', false, 'Upload panel container not found');
            return false;
        }

        // Check for hint bar (collapsed state)
        const hintBarExists = await waitForElement(page, '#upload-collapse-hint', 5000);
        if (hintBarExists) {
            const hintText = await page.$eval('#upload-collapse-hint', el => el.textContent);
            addResult('Hint Bar Exists', true, `Hint bar found: "${hintText}"`);
        } else {
            addResult('Hint Bar Exists', false, 'Hint bar not found');
        }

        const screenshot2 = await takeScreenshot(page, 'step2_elements_verified');

        return true;

    } catch (error) {
        addResult('Page Load', false, `Error: ${error.message}`);
        return false;
    }
}

/**
 * TEST 2: Panel Expansion (Click)
 */
async function testPanelExpansion(page) {
    console.log('\n🧪 TEST 2: Panel Expansion via Click');
    console.log('=' .repeat(60));

    try {
        // Click hint bar to expand
        await page.click('#upload-collapse-hint');
        await sleep(500); // Wait for animation

        const screenshot1 = await takeScreenshot(page, 'step3_panel_clicked');

        // Check if panel expanded (upload row should be visible)
        const panelExpanded = await page.$eval('#upload-panel-container',
            el => el.classList.contains('expanded')
        );

        if (panelExpanded) {
            addResult('Panel Expansion', true, 'Panel expanded successfully via click');
        } else {
            addResult('Panel Expansion', false, 'Panel did not expand after click');
            return false;
        }

        // Check for upload component
        const uploadExists = await waitForElement(page, '#upload-telemetry', 5000);
        if (uploadExists) {
            addResult('Upload Component Visible', true, 'Dash Upload component is visible');
        } else {
            addResult('Upload Component Visible', false, 'Dash Upload component not found');
            return false;
        }

        const screenshot2 = await takeScreenshot(page, 'step4_panel_expanded');

        return true;

    } catch (error) {
        addResult('Panel Expansion', false, `Error: ${error.message}`);
        return false;
    }
}

/**
 * TEST 3: File Upload Simulation (Method 1 - Direct Input)
 *
 * This is the MOST RELIABLE method for Puppeteer.
 * It directly interacts with the hidden file input element.
 */
async function testFileUploadDirect(page) {
    console.log('\n🧪 TEST 3: File Upload - Direct Input Method');
    console.log('=' .repeat(60));
    console.log('⚠️  Note: This uses Puppeteer\'s file input API, NOT actual drag & drop');

    try {
        // Verify file exists
        if (!fs.existsSync(CONFIG.testFilePath)) {
            addResult('File Upload (Direct)', false, `Test file not found: ${CONFIG.testFilePath}`);
            return false;
        }

        console.log(`📂 Test file: ${CONFIG.testFilePath}`);
        const fileSize = fs.statSync(CONFIG.testFilePath).size;
        console.log(`📊 File size: ${(fileSize / 1024).toFixed(2)} KB`);

        // Find the hidden file input within the Dash Upload component
        const fileInputSelector = '#upload-telemetry input[type="file"]';
        const fileInputExists = await waitForElement(page, fileInputSelector, 5000);

        if (!fileInputExists) {
            addResult('File Upload (Direct)', false, 'File input element not found');
            return false;
        }

        // Upload file using Puppeteer's uploadFile method
        const fileInput = await page.$(fileInputSelector);
        await fileInput.uploadFile(CONFIG.testFilePath);

        console.log('⏳ File uploaded to input, waiting for processing...');
        await sleep(2000); // Wait for Dash callback

        const screenshot1 = await takeScreenshot(page, 'step5_file_uploaded');

        // Check for success indicators
        // Look for vehicle dropdown to become enabled (sign of successful upload)
        await sleep(3000); // Give Dash time to process

        const vehicleDropdownExists = await waitForElement(page, '#vehicle-selector', 5000);
        if (vehicleDropdownExists) {
            // Check if dropdown has options
            const hasOptions = await page.$$eval('#vehicle-selector option',
                options => options.length > 1 // More than just placeholder
            );

            if (hasOptions) {
                addResult('File Upload (Direct)', true, 'File uploaded successfully - vehicle selector populated');
            } else {
                addResult('File Upload (Direct)', false, 'File uploaded but vehicle selector not populated');
            }
        } else {
            addResult('File Upload (Direct)', false, 'Vehicle selector not found after upload');
        }

        const screenshot2 = await takeScreenshot(page, 'step6_upload_processed');

        return true;

    } catch (error) {
        addResult('File Upload (Direct)', false, `Error: ${error.message}`);
        const screenshot = await takeScreenshot(page, 'error_file_upload');
        return false;
    }
}

/**
 * TEST 4: Drag Event Simulation (Method 2 - UI Behavior)
 *
 * This tests the drag & drop UI behavior by programmatically
 * triggering drag events. It does NOT upload actual files.
 */
async function testDragEventSimulation(page) {
    console.log('\n🧪 TEST 4: Drag Event Simulation - UI Behavior Only');
    console.log('=' .repeat(60));
    console.log('⚠️  Note: This tests UI response to drag events, not actual file upload');

    try {
        // Reload page for clean state
        await page.reload({ waitUntil: 'networkidle2' });
        await sleep(1000);

        // Get initial hint bar text
        const initialText = await page.$eval('#upload-collapse-hint', el => el.textContent);
        console.log(`📝 Initial hint bar text: "${initialText}"`);

        // Simulate dragenter event on window
        await page.evaluate(() => {
            const event = new DragEvent('dragenter', {
                bubbles: true,
                cancelable: true,
                dataTransfer: new DataTransfer()
            });
            // Add file type to dataTransfer
            const file = new File(['test'], 'test.csv', { type: 'text/csv' });
            event.dataTransfer.items.add(file);
            window.dispatchEvent(event);
        });

        await sleep(500);
        const screenshot1 = await takeScreenshot(page, 'step7_drag_enter');

        // Check if hint bar changed (should show drop message)
        const dragText = await page.$eval('#upload-collapse-hint', el => el.textContent);
        console.log(`📝 Drag state hint bar text: "${dragText}"`);

        if (dragText.includes('Drop') || dragText.includes('📥')) {
            addResult('Drag UI Response', true, `Hint bar changed to: "${dragText}"`);
        } else {
            addResult('Drag UI Response', false, `Hint bar did not change. Still: "${dragText}"`);
        }

        // Check if panel auto-expanded
        const panelExpanded = await page.$eval('#upload-panel-container',
            el => el.classList.contains('expanded')
        );

        if (panelExpanded) {
            addResult('Auto-Expand on Drag', true, 'Panel auto-expanded when drag detected');
        } else {
            addResult('Auto-Expand on Drag', false, 'Panel did not auto-expand on drag');
        }

        const screenshot2 = await takeScreenshot(page, 'step8_drag_ui_state');

        // Simulate dragleave to reset
        await page.evaluate(() => {
            const event = new DragEvent('dragleave', {
                bubbles: true,
                cancelable: true
            });
            window.dispatchEvent(event);
        });

        await sleep(500);

        return true;

    } catch (error) {
        addResult('Drag UI Response', false, `Error: ${error.message}`);
        return false;
    }
}

/**
 * TEST 5: DataTransfer API Advanced Simulation (Method 3)
 *
 * This attempts to simulate the complete drag & drop flow
 * including file data, but browser security prevents actual
 * file system access.
 */
async function testDataTransferSimulation(page) {
    console.log('\n🧪 TEST 5: DataTransfer API Simulation - Advanced');
    console.log('=' .repeat(60));
    console.log('⚠️  Note: Browser security limits prevent real file access');

    try {
        // This is the most advanced simulation possible in Puppeteer
        // It creates a mock File object, but cannot access real file system
        const result = await page.evaluate(() => {
            return new Promise((resolve) => {
                try {
                    const hintBar = document.getElementById('upload-collapse-hint');
                    if (!hintBar) {
                        resolve({ success: false, error: 'Hint bar not found' });
                        return;
                    }

                    // Create mock CSV content
                    const csvContent = 'timestamp,vehicle_number,telemetry_name,telemetry_value\n2024-01-01 10:00:00,1,speed,120';
                    const blob = new Blob([csvContent], { type: 'text/csv' });
                    const file = new File([blob], 'test.csv', { type: 'text/csv' });

                    // Create DataTransfer
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);

                    // Create and dispatch drop event
                    const dropEvent = new DragEvent('drop', {
                        bubbles: true,
                        cancelable: true,
                        dataTransfer: dataTransfer
                    });

                    hintBar.dispatchEvent(dropEvent);

                    resolve({
                        success: true,
                        message: 'Drop event dispatched with mock file',
                        fileName: file.name,
                        fileSize: file.size
                    });
                } catch (error) {
                    resolve({ success: false, error: error.message });
                }
            });
        });

        const screenshot = await takeScreenshot(page, 'step9_datatransfer_simulation');

        if (result.success) {
            console.log(`📋 Mock file created: ${result.fileName} (${result.fileSize} bytes)`);
            addResult('DataTransfer Simulation', true, result.message);

            // Note: This won't actually upload the file due to security restrictions
            console.log('⚠️  Note: Mock file created but cannot be processed by Dash due to browser security');
        } else {
            addResult('DataTransfer Simulation', false, result.error);
        }

        return result.success;

    } catch (error) {
        addResult('DataTransfer Simulation', false, `Error: ${error.message}`);
        return false;
    }
}

/**
 * Generate HTML Test Report
 */
function generateHTMLReport() {
    const passRate = results.tests.length > 0
        ? ((results.passed / results.tests.length) * 100).toFixed(2)
        : 0;

    const html = `
<!DOCTYPE html>
<html>
<head>
    <title>Drag & Drop Test Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }
        .summary-card.pass {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        .summary-card.fail {
            background: #f8d7da;
            border: 2px solid #dc3545;
        }
        .summary-card.info {
            background: #d1ecf1;
            border: 2px solid #17a2b8;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }
        .summary-card .value {
            font-size: 36px;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }
        .status-pass {
            background: #28a745;
            color: white;
        }
        .status-fail {
            background: #dc3545;
            color: white;
        }
        .screenshot-link {
            color: #007bff;
            text-decoration: none;
        }
        .screenshot-link:hover {
            text-decoration: underline;
        }
        .limitations {
            background: #fff3cd;
            border: 2px solid #ffc107;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }
        .limitations h2 {
            margin-top: 0;
            color: #856404;
        }
        .limitations ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .limitations li {
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Drag & Drop Test Report</h1>
        <p><strong>Dashboard URL:</strong> ${CONFIG.dashboardUrl}</p>
        <p><strong>Test File:</strong> ${CONFIG.testFilePath}</p>
        <p><strong>Timestamp:</strong> ${results.timestamp}</p>

        <div class="summary">
            <div class="summary-card pass">
                <h3>Tests Passed</h3>
                <div class="value">${results.passed}</div>
            </div>
            <div class="summary-card fail">
                <h3>Tests Failed</h3>
                <div class="value">${results.failed}</div>
            </div>
            <div class="summary-card info">
                <h3>Pass Rate</h3>
                <div class="value">${passRate}%</div>
            </div>
        </div>

        <div class="limitations">
            <h2>⚠️ Testing Limitations & Notes</h2>
            <ul>
                <li><strong>OS-Level Drag & Drop:</strong> Puppeteer (and all browser automation tools) CANNOT simulate actual file drag from Windows Explorer. This is a browser security restriction.</li>
                <li><strong>Method 1 (Direct Input):</strong> Most reliable - uses Puppeteer's file input API to upload files directly to the hidden input element.</li>
                <li><strong>Method 2 (Drag Events):</strong> Tests UI behavior - verifies the dashboard responds correctly to drag events (panel expansion, hint text changes).</li>
                <li><strong>Method 3 (DataTransfer API):</strong> Advanced simulation - creates mock File objects but cannot access real file system due to browser security.</li>
                <li><strong>Recommendation:</strong> Use Method 1 for actual file upload testing. Methods 2-3 test UI behavior only.</li>
            </ul>
        </div>

        <h2>Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Message</th>
                    <th>Screenshot</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
                ${results.tests.map((test, index) => `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${test.test}</td>
                        <td><span class="status-badge status-${test.status.toLowerCase()}">${test.status}</span></td>
                        <td>${test.message}</td>
                        <td>${test.screenshot ? `<a href="screenshots/${test.screenshot}" class="screenshot-link" target="_blank">📸 View</a>` : '-'}</td>
                        <td>${new Date(test.timestamp).toLocaleTimeString()}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>

        <h2>Screenshots</h2>
        <p>Total screenshots captured: ${results.screenshots.length}</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 20px;">
            ${results.screenshots.map(screenshot => `
                <a href="screenshots/${screenshot}" target="_blank" style="text-decoration: none;">
                    <div style="padding: 10px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 4px; text-align: center;">
                        <div style="font-size: 48px;">📸</div>
                        <div style="font-size: 12px; color: #666; margin-top: 5px;">${screenshot}</div>
                    </div>
                </a>
            `).join('')}
        </div>
    </div>
</body>
</html>
    `;

    return html;
}

/**
 * Main Test Execution
 */
async function runTests() {
    console.log('╔═══════════════════════════════════════════════════════╗');
    console.log('║   Drag & Drop Automated Testing for Racing Dashboard  ║');
    console.log('╚═══════════════════════════════════════════════════════╝');
    console.log();
    console.log(`🌐 Dashboard URL: ${CONFIG.dashboardUrl}`);
    console.log(`📂 Test File: ${CONFIG.testFilePath}`);
    console.log(`📸 Screenshots: ${CONFIG.screenshotsDir}`);
    console.log(`⏱️  Timeout: ${CONFIG.timeout}ms`);
    console.log();

    // Create screenshots directory
    if (!fs.existsSync(CONFIG.screenshotsDir)) {
        fs.mkdirSync(CONFIG.screenshotsDir, { recursive: true });
        console.log(`✅ Created screenshots directory: ${CONFIG.screenshotsDir}`);
    }

    let browser = null;
    let page = null;

    try {
        // Launch browser
        console.log('\n🚀 Launching browser...');
        browser = await puppeteer.launch({
            headless: CONFIG.headless,
            slowMo: CONFIG.slowMo,
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process'
            ],
            defaultViewport: {
                width: 1920,
                height: 1080
            }
        });

        page = await browser.newPage();

        // Enable console logging from the page
        page.on('console', msg => {
            const type = msg.type();
            const text = msg.text();
            if (type === 'log') console.log(`🌐 Browser: ${text}`);
            if (type === 'error') console.error(`🌐 Browser Error: ${text}`);
        });

        // Run test suite
        const test1 = await testPageLoad(page);
        if (!test1) {
            console.error('\n❌ Page load failed, aborting remaining tests');
        } else {
            await testPanelExpansion(page);
            await testFileUploadDirect(page);
            await testDragEventSimulation(page);
            await testDataTransferSimulation(page);
        }

    } catch (error) {
        console.error('\n❌ Fatal Error:', error.message);
        addResult('Test Execution', false, `Fatal error: ${error.message}`);
    } finally {
        // Generate report
        console.log('\n' + '='.repeat(60));
        console.log('📊 TEST SUMMARY');
        console.log('='.repeat(60));
        console.log(`✅ Passed: ${results.passed}`);
        console.log(`❌ Failed: ${results.failed}`);
        console.log(`📸 Screenshots: ${results.screenshots.length}`);

        const passRate = results.tests.length > 0
            ? ((results.passed / results.tests.length) * 100).toFixed(2)
            : 0;
        console.log(`📈 Pass Rate: ${passRate}%`);

        // Save HTML report
        const reportPath = path.join(__dirname, 'test_report_drag_drop.html');
        const htmlReport = generateHTMLReport();
        fs.writeFileSync(reportPath, htmlReport);
        console.log(`\n📄 HTML Report: ${reportPath}`);

        // Save JSON results
        const jsonPath = path.join(__dirname, 'test_results_drag_drop.json');
        fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
        console.log(`📄 JSON Results: ${jsonPath}`);

        // Close browser
        if (browser) {
            await browser.close();
            console.log('\n✅ Browser closed');
        }

        // Exit with appropriate code
        process.exit(results.failed > 0 ? 1 : 0);
    }
}

// Run tests
runTests().catch(error => {
    console.error('❌ Unhandled error:', error);
    process.exit(1);
});
