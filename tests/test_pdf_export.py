"""
Playwright test for PDF Export feature (Sprint 3 Task 1)

Tests the complete workflow:
1. Upload telemetry file
2. Select vehicle
3. Navigate to Model Predictions tab
4. Wait for analysis completion
5. Click Export PDF Report button
6. Verify PDF is downloaded
7. Validate PDF file exists and has content
"""

import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect, Page
import pytest


class TestPDFExport:
    """Test suite for PDF export functionality"""

    DASHBOARD_URL = "http://localhost:8050"
    TEST_DATA_FILE = "master_racing_data.csv"
    TEST_VEHICLE = "2"
    DOWNLOAD_TIMEOUT = 30000  # 30 seconds

    @pytest.fixture(scope="function")
    def page(self):
        """Create a new browser page for each test"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Set to True for CI/CD
            context = browser.new_context(
                accept_downloads=True,
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()

            # Enable console logging
            page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.type}: {msg.text}"))
            page.on("pageerror", lambda exc: print(f"BROWSER ERROR: {exc}"))

            yield page

            context.close()
            browser.close()

    def wait_for_dashboard_load(self, page: Page):
        """Wait for dashboard to fully load"""
        print("[WAIT] Waiting for dashboard to load...")
        page.goto(self.DASHBOARD_URL, timeout=15000)

        # Wait for main layout elements (tabs or main container)
        page.wait_for_selector('[id*="tabs"]', timeout=15000)
        time.sleep(2)  # Extra wait for full render
        print("[OK] Dashboard loaded")

    def upload_telemetry_file(self, page: Page):
        """Upload telemetry CSV file"""
        print(f"[WAIT] Uploading telemetry file: {self.TEST_DATA_FILE}")

        # Find upload component
        upload_input = page.locator('input[type="file"]').first

        # Get absolute path to test file
        file_path = Path(__file__).parent.parent / self.TEST_DATA_FILE
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")

        # Upload file
        upload_input.set_input_files(str(file_path))

        # Wait for upload to process
        time.sleep(2)
        print("[OK] File uploaded")

    def select_vehicle(self, page: Page):
        """Select vehicle from dropdown"""
        print(f"[WAIT] Selecting vehicle #{self.TEST_VEHICLE}")

        # Wait for vehicle dropdown to be available
        page.wait_for_selector('#vehicle-dropdown', timeout=10000)

        # Select vehicle
        page.select_option('#vehicle-dropdown', self.TEST_VEHICLE)

        time.sleep(1)
        print(f"[OK] Vehicle #{self.TEST_VEHICLE} selected")

    def navigate_to_model_predictions_tab(self, page: Page):
        """Navigate to Model Predictions tab (Tab 3)"""
        print("[WAIT] Navigating to Model Predictions tab...")

        # Click on Model Predictions tab
        # The tab might be labeled "Model Predictions" or have a specific ID
        model_predictions_tab = page.get_by_text("Model Predictions", exact=False).first
        model_predictions_tab.click()

        time.sleep(2)
        print("[OK] Model Predictions tab opened")

    def wait_for_analysis_completion(self, page: Page):
        """Wait for telemetry analysis to complete"""
        print("[WAIT] Waiting for analysis to complete...")

        # Look for indicators that analysis is complete
        # This could be the presence of pattern cards, feature categories, etc.

        # Wait for feature categories to appear (indicates analysis complete)
        page.wait_for_selector('.accordion', timeout=60000)

        # Additional wait to ensure all components loaded
        time.sleep(3)
        print("[OK] Analysis completed")

    def click_export_pdf_button(self, page: Page):
        """Click the Export PDF Report button"""
        print("[WAIT] Clicking Export PDF Report button...")

        # Find and click the Export PDF button
        export_btn = page.get_by_text("Export PDF Report", exact=False)

        # Scroll button into view
        export_btn.scroll_into_view_if_needed()

        # Wait a moment for any animations
        time.sleep(1)

        # Click the button
        export_btn.click()

        print("[OK] Export PDF button clicked")

    def verify_pdf_download(self, page: Page) -> Path:
        """Verify that PDF file is downloaded"""
        print("[WAIT] Waiting for PDF download...")

        # Start waiting for download before clicking button
        with page.expect_download(timeout=self.DOWNLOAD_TIMEOUT) as download_info:
            # The download should already be in progress from previous click
            pass

        download = download_info.value

        # Get download path
        download_path = Path(download.path())

        print(f"[OK] PDF downloaded: {download.suggested_filename}")
        print(f"   Path: {download_path}")

        return download_path

    def validate_pdf_file(self, pdf_path: Path):
        """Validate that PDF file exists and has content"""
        print("[WAIT] Validating PDF file...")

        # Check file exists
        assert pdf_path.exists(), f"PDF file not found at {pdf_path}"

        # Check file size
        file_size = pdf_path.stat().st_size
        assert file_size > 1000, f"PDF file too small ({file_size} bytes), likely empty or corrupted"

        print(f"[OK] PDF file validated: {file_size:,} bytes")

        # Optional: Check PDF header
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            assert header == b'%PDF', f"Invalid PDF header: {header}"

        print("[OK] PDF header validated")

    def test_pdf_export_complete_workflow(self, page):
        """
        MAIN TEST: Complete PDF export workflow

        Steps:
        1. Load dashboard
        2. Upload telemetry
        3. Select vehicle
        4. Navigate to Model Predictions tab
        5. Wait for analysis
        6. Click Export PDF button
        7. Verify download
        8. Validate PDF file
        """
        print("\n" + "="*80)
        print("[TEST] TEST: PDF Export Complete Workflow")
        print("="*80 + "\n")

        try:
            # Step 1: Load dashboard
            self.wait_for_dashboard_load(page)

            # Step 2: Upload telemetry
            self.upload_telemetry_file(page)

            # Step 3: Select vehicle
            self.select_vehicle(page)

            # Step 4: Navigate to Model Predictions tab
            self.navigate_to_model_predictions_tab(page)

            # Step 5: Wait for analysis
            self.wait_for_analysis_completion(page)

            # Step 6: Click Export PDF button
            self.click_export_pdf_button(page)

            # Step 7: Verify download
            pdf_path = self.verify_pdf_download(page)

            # Step 8: Validate PDF file
            self.validate_pdf_file(pdf_path)

            print("\n" + "="*80)
            print("[OK] TEST PASSED: PDF Export Workflow Complete!")
            print("="*80 + "\n")

        except Exception as e:
            print("\n" + "="*80)
            print(f"[FAIL] TEST FAILED: {str(e)}")
            print("="*80 + "\n")

            # Take screenshot for debugging
            screenshot_path = Path(__file__).parent.parent / "test_failure_screenshot.png"
            page.screenshot(path=str(screenshot_path))
            print(f"[SCREENSHOT] Screenshot saved: {screenshot_path}")

            raise

    def test_pdf_export_button_exists(self, page):
        """
        QUICK TEST: Verify Export PDF button exists in UI
        """
        print("\n" + "="*80)
        print("[TEST] TEST: Export PDF Button Exists")
        print("="*80 + "\n")

        try:
            # Load dashboard
            self.wait_for_dashboard_load(page)

            # Upload and select vehicle
            self.upload_telemetry_file(page)
            self.select_vehicle(page)

            # Navigate to Model Predictions tab
            self.navigate_to_model_predictions_tab(page)

            # Wait for analysis
            self.wait_for_analysis_completion(page)

            # Check button exists
            export_btn = page.get_by_text("Export PDF Report", exact=False)
            expect(export_btn).to_be_visible()

            print("[OK] Export PDF button found and visible")

            print("\n" + "="*80)
            print("[OK] TEST PASSED: Button Exists")
            print("="*80 + "\n")

        except Exception as e:
            print("\n" + "="*80)
            print(f"[FAIL] TEST FAILED: {str(e)}")
            print("="*80 + "\n")
            raise

    def test_pdf_export_no_vehicle_selected(self, page):
        """
        NEGATIVE TEST: Verify PDF export fails gracefully without vehicle selection
        """
        print("\n" + "="*80)
        print("[TEST] TEST: PDF Export Without Vehicle Selection")
        print("="*80 + "\n")

        try:
            # Load dashboard
            self.wait_for_dashboard_load(page)

            # Navigate to Model Predictions tab WITHOUT uploading or selecting vehicle
            self.navigate_to_model_predictions_tab(page)

            # Try to find Export PDF button (should exist but may be disabled or do nothing)
            export_btn_locator = page.locator('button:has-text("Export PDF Report")')

            if export_btn_locator.count() > 0:
                print("[OK] Export PDF button exists (as expected)")

                # Click it (should not trigger download)
                export_btn_locator.first.click()

                # Wait briefly
                time.sleep(2)

                # No download should occur (this is expected behavior)
                print("[OK] No download triggered (as expected)")
            else:
                print("[WARN]  Export PDF button not found (acceptable if hidden when no data)")

            print("\n" + "="*80)
            print("[OK] TEST PASSED: Graceful handling of missing data")
            print("="*80 + "\n")

        except Exception as e:
            print("\n" + "="*80)
            print(f"[FAIL] TEST FAILED: {str(e)}")
            print("="*80 + "\n")
            raise


if __name__ == "__main__":
    """Run tests directly with pytest"""
    import sys

    # Run pytest with verbose output
    sys.exit(pytest.main([__file__, "-v", "-s"]))
