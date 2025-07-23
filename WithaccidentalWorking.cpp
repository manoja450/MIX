///working, but total Michel count is less
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TMath.h>
#include <TMarker.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TGaxis.h>
#include <TLine.h>
#include <TLatex.h>
#include <TPaveText.h>
#include <TMultiGraph.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TAxis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>

using std::cout;
using std::endl;
using std::cerr;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const int PULSE_THRESHOLD = 30;     // ADC threshold for pulse detection
const int BS_UNCERTAINTY = 5;       // Baseline uncertainty (ADC)
const int EV61_THRESHOLD = 1200;    // Beam on if channel 22 > this (ADC)
const double MUON_ENERGY_THRESHOLD = 50; // Min PMT energy for muon (p.e.)
const double MICHEL_ENERGY_MIN = 40;    // Min PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX = 1000;  // Max PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX_DT = 400; // Max PMT energy for dt plots (p.e.)
const double MICHEL_DT_MIN = 0.76;      // Min time after muon for Michel (µs)
const double MICHEL_DT_MAX = 16.0;      // Max time after muon for Michel (µs)
const int ADCSIZE = 45;                 // Number of ADC samples per waveform
const std::vector<double> SIDE_VP_THRESHOLDS = {750, 950, 1200, 1375, 525, 700, 700, 500}; // Channels 12-19 (ADC)
const double TOP_VP_THRESHOLD = 450;    // Channels 20-21 (ADC)
const double FIT_MIN = 1.0;             // Fit range min (µs)
const double FIT_MAX = 16.0;            // Fit range max (µs)
const double MICHEL_PEAK_FIT_MIN = 100; // Min for Michel peak fit (p.e.)
const double MICHEL_PEAK_FIT_MAX = 300; // Max for Michel peak fit (p.e.)

// Global variable for accidental background time constant
double FIXED_TAU_ACCIDENTAL = 0.0;

// Generate unique output directory with timestamp
string getTimestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", t);
    return string(buffer);
}
const string OUTPUT_DIR = "./AnalysisOutput_" + getTimestamp();

// Pulse structure
struct pulse {
    double start;          // Start time (µs)
    double end;            // End time (µs)
    double peak;           // Max amplitude (p.e. for PMTs, ADC for SiPMs)
    double energy;         // Energy (p.e. for PMTs, ADC for SiPMs)
    double number;         // Number of channels with pulse
    bool single;           // Timing consistency
    bool beam;             // Beam status
    double trigger;        // Trigger type
    double side_vp_energy; // Side veto energy (ADC)
    double top_vp_energy;  // Top veto energy (ADC)
    double all_vp_energy;  // All veto energy (ADC)
    double last_muon_time; // Time of last muon (µs)
    bool is_muon;          // Muon candidate flag
    bool is_michel;        // Michel electron candidate flag
};

// Temporary pulse structure
struct pulse_temp {
    double start;  // Start time (µs)
    double end;    // End time (µs)
    double peak;   // Max amplitude
    double energy; // Energy
};

// Helper function to find maximum bin in a range
int getMaxBinInRange(TH1* hist, int bin_min, int bin_max) {
    double max_val = -1e9;
    int max_bin = bin_min;
    for (int i = bin_min; i <= bin_max; i++) {
        double val = hist->GetBinContent(i);
        if (val > max_val) {
            max_val = val;
            max_bin = i;
        }
    }
    return max_bin;
}

// Fitting functions for SPE calibration
Double_t fitGauss(Double_t *x, Double_t *par) {
    return par[0] * TMath::Gaus(x[0], par[1], par[2]);
}

Double_t six_fit_func(Double_t *x, Double_t *par) {
    return (par[0] * TMath::Gaus(x[0], par[1], par[2]) + 
           par[3] * TMath::Gaus(x[0], par[4], par[5]));
}

Double_t eight_fit_func(Double_t *x, Double_t *par) {
    return (par[0] * TMath::Gaus(x[0], par[1], par[2]) + 
           par[3] * TMath::Gaus(x[0], par[4], par[5]) + 
           par[6] * TMath::Gaus(x[0], 2.0 * par[4], TMath::Sqrt(2.0 * par[5]*par[5] - par[2]*par[2])) + 
           par[7] * TMath::Gaus(x[0], 3.0 * par[4], TMath::Sqrt(3.0 * par[5]*par[5] - 2.0 * par[2]*par[2])));
}

// Double exponential fit function: N0*exp(-t/τ_michel) + N1*exp(-t/τ_accidental)
Double_t DoubleExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]) + par[2] * exp(-x[0]/par[3]);
}

// Constrained double exponential fit (with fixed τ_accidental)
Double_t ConstrainedDoubleExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]) + par[2] * exp(-x[0]/FIXED_TAU_ACCIDENTAL);
}

//  exponential fit for fit start comparison
Double_t SimpleExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]) + par[2] * exp(-x[0]/FIXED_TAU_ACCIDENTAL);
}

// Individual exponential components for visualization
Double_t MichelExp(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]);
}

Double_t AccidentalExp(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]);
}

// Function to set consistent plot style
void setPlotStyle(TH1* hist) {
    hist->SetLineWidth(2);
    hist->GetXaxis()->SetTitleSize(0.045);
    hist->GetYaxis()->SetTitleSize(0.045);
    hist->GetXaxis()->SetLabelSize(0.04);
    hist->GetYaxis()->SetLabelSize(0.04);
    hist->GetYaxis()->SetTitleOffset(1.2);
}

// Function to set smart y-axis range
void setSmartYRange(TH1* hist, double min_frac = 0.1, double max_frac = 1.1) {
    double ymax = hist->GetBinContent(hist->GetMaximumBin()) * max_frac;
    double ymin = -hist->GetBinContent(hist->GetMaximumBin()) * min_frac;
    hist->SetMaximum(ymax);
    hist->SetMinimum(ymin);
}

// Function to check if an event meets Michel-like criteria (energy, veto, trigger)
bool isMichelLike(double energy, const std::vector<double>& veto_energies, int triggerBits) {
    // Check energy range
    if (energy < MICHEL_ENERGY_MIN || energy > MICHEL_ENERGY_MAX) return false;
    
    // Check veto system
    bool veto_low = true;
    for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
        if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
            veto_low = false;
            break;
        }
    }
    if (veto_energies[8] > TOP_VP_THRESHOLD || veto_energies[9] > TOP_VP_THRESHOLD) {
        veto_low = false;
    }
    if (!veto_low) return false;
    
    // Check trigger
    if (triggerBits == 1 || triggerBits == 4 || triggerBits == 8 || triggerBits == 16) return false;
    
    return true;
}

// Function to plot fit start time comparison
void plotFitStartComparison(TH1D* h_dt_michel) {
    if (h_dt_michel->GetEntries() < 100) {
        cout << "Warning: Insufficient entries for fit start comparison (" << h_dt_michel->GetEntries() << ")" << endl;
        return;
    }

    // Prepare vectors to store results
    vector<double> start_times, chi2_ndf, tau_values, tau_errors;
    
    // Perform fits with different start times
    for (double start = 1.0; start <= 4.0; start += 0.5) {
        TF1* f_fit = new TF1("f_fit", SimpleExpFit, start, FIT_MAX, 2);
        f_fit->SetParNames("N0", "#tau");
        f_fit->SetParameters(h_dt_michel->GetBinContent(h_dt_michel->FindBin(start)), 2.2);
        f_fit->SetParLimits(0, 0, 1e6);
        f_fit->SetParLimits(1, 0.1, 10.0);
        
        h_dt_michel->Fit(f_fit, "RQN", "", start, FIT_MAX);
        
        start_times.push_back(start);
        chi2_ndf.push_back(f_fit->GetChisquare() / f_fit->GetNDF());
        tau_values.push_back(f_fit->GetParameter(1));
        tau_errors.push_back(f_fit->GetParError(1));
        
        delete f_fit;
    }

    // Find best fit start time (minimum chi2/NDF)
    auto min_chi2_it = min_element(chi2_ndf.begin(), chi2_ndf.end());
    int best_index = distance(chi2_ndf.begin(), min_chi2_it);
    double best_start = start_times[best_index];
    double best_tau = tau_values[best_index];
    double best_tau_err = tau_errors[best_index];
    
    // Create the comparison plot
    TCanvas* c_compare = new TCanvas("c_compare", "Fit Start Comparison", 1200, 800);
    c_compare->SetMargin(0.12, 0.05, 0.12, 0.08);
    
    // Create pad for the plot
    TPad* pad = new TPad("pad", "pad", 0, 0, 1, 1);
    pad->Draw();
    pad->cd();
    
    // Find ranges for axes
    double chi2_min = *min_element(chi2_ndf.begin(), chi2_ndf.end());
    double chi2_max = *max_element(chi2_ndf.begin(), chi2_ndf.end());
    double tau_min = *min_element(tau_values.begin(), tau_values.end());
    double tau_max = *max_element(tau_values.begin(), tau_values.end());
    
    // Scale factors for right axis
    double scale_min = chi2_min;
    double scale_max = chi2_max;
    double tau_scale_min = tau_min;
    double tau_scale_max = tau_max;
    
    // Scale tau values to match chi2/NDF range
    vector<double> scaled_tau_values;
    for (size_t i = 0; i < tau_values.size(); i++) {
        scaled_tau_values.push_back(scale_min + (tau_values[i] - tau_scale_min) * (scale_max - scale_min) / (tau_scale_max - tau_scale_min));
    }
    
    // Create graphs
    TGraph* g_chi2 = new TGraph(start_times.size(), &start_times[0], &chi2_ndf[0]);
    g_chi2->SetTitle("#chi^{2}/NDF");
    g_chi2->SetMarkerStyle(20);
    g_chi2->SetMarkerColor(kBlue);
    g_chi2->SetLineColor(kBlue);
    g_chi2->SetLineWidth(2);
    
    TGraph* g_tau = new TGraph(start_times.size(), &start_times[0], &scaled_tau_values[0]);
    g_tau->SetTitle("#tau");
    g_tau->SetMarkerStyle(21);
    g_tau->SetMarkerColor(kRed);
    g_tau->SetLineColor(kRed);
    g_tau->SetLineWidth(2);
    
    // Create multi-graph
    TMultiGraph* mg = new TMultiGraph();
    mg->Add(g_chi2);
    mg->Add(g_tau);
    mg->Draw("APL");
    
    // Set axis titles
    mg->GetXaxis()->SetTitle("Fit Start Time (#mus)");
    mg->GetXaxis()->SetTitleSize(0.045);
    mg->GetXaxis()->SetLabelSize(0.04);
    mg->GetYaxis()->SetTitle("#chi^{2}/NDF");
    mg->GetYaxis()->SetTitleSize(0.045);
    mg->GetYaxis()->SetLabelSize(0.04);
    mg->GetYaxis()->SetTitleOffset(1.2);
    
    // Add right axis for tau
    c_compare->Update();
    double ymin = pad->GetUymin();
    double ymax = pad->GetUymax();
    
    TGaxis* axis = new TGaxis(gPad->GetUxmax(), ymin, 
                             gPad->GetUxmax(), ymax,
                             tau_scale_min, tau_scale_max, 510, "+L");
    axis->SetLineColor(kRed);
    axis->SetLabelColor(kRed);
    axis->SetTitleColor(kRed);
    axis->SetTitle("#tau (#mus)");
    axis->SetTitleSize(0.045);
    axis->SetLabelSize(0.04);
    axis->SetTitleOffset(1.2);
    axis->Draw();
    
    // Add legend
    TLegend* leg = new TLegend(0.6, 0.7, 0.88, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);
    leg->AddEntry(g_chi2, "#chi^{2}/NDF", "lp");
    leg->AddEntry(g_tau, "#tau (#mus)", "lp");
    leg->Draw();
    
    // Mark best fit point
    TMarker* m_best = new TMarker(best_start, chi2_ndf[best_index], 29);
    m_best->SetMarkerSize(2);
    m_best->SetMarkerColor(kGreen+2);
    m_best->Draw();
    
    c_compare->Update();
    string plotName = OUTPUT_DIR + "/FitStartComparison.png";
    c_compare->SaveAs(plotName.c_str());
    cout << "Saved fit start comparison plot: " << plotName << endl;
    
    // Cleanup
    delete c_compare;
    delete pad;
    delete mg;
    delete g_chi2;
    delete g_tau;
    delete axis;
    delete leg;
    delete m_best;
}

// Function to fit Michel energy peak
double fitMichelPeak(TH1D* hist) {
    if (hist->GetEntries() < 100) {
        cout << "Warning: Insufficient entries in Michel energy histogram (" << hist->GetEntries() << ")" << endl;
        return 0.0;
    }

    // Create and configure the fit function
    TF1* f_gaus = new TF1("f_gaus", fitGauss, MICHEL_PEAK_FIT_MIN, MICHEL_PEAK_FIT_MAX, 3);
    f_gaus->SetParameters(hist->GetMaximum(), 200, 50);
    f_gaus->SetParNames("A0", "#mu", "#sigma");
    f_gaus->SetParLimits(1, MICHEL_PEAK_FIT_MIN, MICHEL_PEAK_FIT_MAX);
    f_gaus->SetParLimits(2, 10, 100);
    
    // Perform the fit
    hist->Fit(f_gaus, "RQ", "", MICHEL_PEAK_FIT_MIN, MICHEL_PEAK_FIT_MAX);
    double peak = f_gaus->GetParameter(1);
    double peak_err = f_gaus->GetParError(1);
    cout << "Michel peak: " << peak << " ± " << peak_err << " p.e." << endl;

    // Create canvas and set style
    TCanvas* c_michel_fit = new TCanvas("c_michel_fit", "Michel Energy Fit", 1200, 800);
    c_michel_fit->SetMargin(0.12, 0.05, 0.12, 0.08);
    
    // Set histogram style
    setPlotStyle(hist);
    hist->SetLineColor(kRed);
    hist->SetTitle("Michel Electron Energy Distribution;Energy (p.e.);Counts / 8 p.e.");
    
    // Set smart y-axis range
    int bin_min = hist->FindBin(MICHEL_PEAK_FIT_MIN);
    int bin_max = hist->FindBin(MICHEL_PEAK_FIT_MAX);
    int max_bin = getMaxBinInRange(hist, bin_min, bin_max);
    double ymax_data = hist->GetBinContent(max_bin);
    double ymax = ymax_data * 1.1;
    hist->SetMaximum(ymax);
    
    // Draw the histogram
    hist->Draw("HIST");
    
    // Create a clone of the fit function limited to the fit range
    TF1* f_gaus_clone = new TF1(*f_gaus);
    f_gaus_clone->SetRange(MICHEL_PEAK_FIT_MIN, MICHEL_PEAK_FIT_MAX);
    f_gaus_clone->SetLineColor(kBlue);
    f_gaus_clone->SetLineWidth(3);
    f_gaus_clone->Draw("same");
   
    
    c_michel_fit->Update();
    string plotName = OUTPUT_DIR + "/Michel_Energy_Fit.png";
    c_michel_fit->SaveAs(plotName.c_str());
    cout << "Saved Michel energy fit plot: " << plotName << endl;
    
    // Cleanup
    delete c_michel_fit;
    delete f_gaus;
    delete f_gaus_clone;
   
    
    return peak;
}

// Function to plot Michel energy 
void plotMichelEnergy(TH1D* h_michel_energy) {
    TCanvas *c_energy = new TCanvas("c_energy", "Michel Energy", 1200, 800);
    c_energy->SetMargin(0.12, 0.05, 0.12, 0.08);
    
    // Set histogram style
    setPlotStyle(h_michel_energy);
    h_michel_energy->SetLineColor(kRed);
    h_michel_energy->SetTitle("Michel Electron Energy Distribution;Energy (p.e.);Counts / 8 p.e.");
    
    // Set smart y-axis range
    int bin_min = h_michel_energy->FindBin(MICHEL_ENERGY_MIN);
    int bin_max = h_michel_energy->FindBin(MICHEL_ENERGY_MAX);
    int max_bin = getMaxBinInRange(h_michel_energy, bin_min, bin_max);
    double ymax_data = h_michel_energy->GetBinContent(max_bin);
    double ymax = ymax_data * 1.1;
    h_michel_energy->SetMaximum(ymax);
    
    h_michel_energy->Draw("HIST");

    c_energy->Update();
    string plotName = OUTPUT_DIR + "/Michel_Energy.png";
    c_energy->SaveAs(plotName.c_str());
    cout << "Saved Michel energy plot: " << plotName << endl;
    
    delete c_energy;
}

// Function to fit Michel dt distribution with proper visualization
void fitMichelDt(TH1D* h_dt_michel) {
    if (h_dt_michel->GetEntries() < 50) {
        cout << "Warning: Insufficient entries in Michel dt histogram (" << h_dt_michel->GetEntries() << ")" << endl;
        return;
    }

    TCanvas *c_dt = new TCanvas("c_dt", "Michel dt Fit", 1200, 800);
    c_dt->SetMargin(0.12, 0.05, 0.12, 0.08);
    
    // Set histogram style
    setPlotStyle(h_dt_michel);
    h_dt_michel->SetLineColor(kBlack);
    h_dt_michel->SetTitle("Muon-Michel Time Difference;Time to Previous Muon (#mus);Counts / 0.08 #mus");
    
    // Set smart y-axis range
    double ymax_data = h_dt_michel->GetBinContent(h_dt_michel->GetMaximumBin());
    double ymax = ymax_data * 1.1;
    h_dt_michel->SetMaximum(ymax);
    
    h_dt_michel->Draw("HIST");

    TF1* f_michel = new TF1("f_michel", ConstrainedDoubleExpFit, FIT_MIN, FIT_MAX, 3);
    f_michel->SetParNames("N_{0}", "#tau_{#mu}", "N_{acc}");
    
    double N0_init = h_dt_michel->GetBinContent(h_dt_michel->FindBin(FIT_MIN));
    double N_acc_init = h_dt_michel->GetBinContent(h_dt_michel->FindBin(FIT_MAX));
    f_michel->SetParameters(N0_init, 2.2, N_acc_init);
    
    f_michel->SetParLimits(0, 0, 1e6);
    f_michel->SetParLimits(1, 0.1, 10.0);
    f_michel->SetParLimits(2, 0, 1e6);
    
    f_michel->SetLineColor(kRed);
    f_michel->SetLineWidth(3);
    f_michel->SetNpx(1000);

    int fitStatus = h_dt_michel->Fit(f_michel, "R");
    if (fitStatus != 0) {
        cout << "Warning: Fit failed to converge (status " << fitStatus << ")" << endl;
    } else {
        TF1* f_michel_comp = new TF1("f_michel_comp", MichelExp, FIT_MIN, FIT_MAX, 2);
        f_michel_comp->SetParameters(f_michel->GetParameter(0), f_michel->GetParameter(1));
        f_michel_comp->SetLineColor(kBlue);
        f_michel_comp->SetLineWidth(2);
        f_michel_comp->SetLineStyle(2);
        f_michel_comp->Draw("same");

        f_michel->Draw("same");

    }

    c_dt->Update();
    string plotName = OUTPUT_DIR + "/Michel_dt.png";
    c_dt->SaveAs(plotName.c_str());
    cout << "Saved Michel dt plot: " << plotName << endl;
    
    // Perform fit start comparison
    plotFitStartComparison(h_dt_michel);
    
    delete c_dt;
}

// Function to determine accidental background time constant and energy spectrum
double determineAccidentalTau(const vector<string>& inputFiles, const Double_t* mu1) {
    TH1D* h_accidental_dt = new TH1D("h_accidental_dt", 
        "Accidental Background dt;dt (#mus);Counts/0.08 #mus", 
        200, 0, MICHEL_DT_MAX);
    TH1D* h_accidental_energy = new TH1D("h_accidental_energy",
        "Accidental Background Energy;Energy (p.e.);Counts/8 p.e.",
        100, 0, 800);

    double first_event_time = -1;
    int num_accidental = 0;
    std::vector<double> veto_energies(10, 0);
    std::vector<double> event_times;
    std::vector<bool> is_muon_event;

    // First pass: collect event times and muon flags from all input files
    for (const auto& inputFileName : inputFiles) {
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile* file = TFile::Open(inputFileName.c_str());
        if (!file || file->IsZombie()) {
            cerr << "Error opening input file: " << inputFileName << endl;
            continue;
        }

        TTree* tree = (TTree*)file->Get("tree");
        if (!tree) {
            cerr << "Error accessing tree in input file: " << inputFileName << endl;
            file->Close();
            delete file;
            continue;
        }

        // Variables for tree branches
        Int_t eventID, nSamples[23], peakPosition[23], triggerBits;
        Short_t adcVal[23][45];
        Double_t baselineMean[23], baselineRMS[23], pulseH[23], area[23];
        Long64_t nsTime;

        // Set branch addresses
        tree->SetBranchAddress("eventID", &eventID);
        tree->SetBranchAddress("nSamples", nSamples);
        tree->SetBranchAddress("adcVal", adcVal);
        tree->SetBranchAddress("baselineMean", baselineMean);
        tree->SetBranchAddress("baselineRMS", baselineRMS);
        tree->SetBranchAddress("pulseH", pulseH);
        tree->SetBranchAddress("peakPosition", peakPosition);
        tree->SetBranchAddress("area", area);
        tree->SetBranchAddress("nsTime", &nsTime);
        tree->SetBranchAddress("triggerBits", &triggerBits);

        // Process each entry in the current file
        for (int iEnt = 0; iEnt < tree->GetEntries(); iEnt++) {
            tree->GetEntry(iEnt);
            
            double current_time = nsTime / 1000.0; // ns to µs
            if (first_event_time < 0) first_event_time = current_time;
            
            // Calculate event energy
            double event_energy = 0;
            for (int iChan = 0; iChan < N_PMTS; iChan++) {
                if (mu1[iChan] > 0) event_energy += area[PMT_CHANNEL_MAP[iChan]] / mu1[iChan];
            }
            
            // Check if this is a muon event
            bool is_muon = false;
            for (int iChan = 12; iChan < 20; iChan++) {
                veto_energies[iChan - 12] = area[iChan];
            }
            veto_energies[8] = area[20] * 1.07809;
            veto_energies[9] = area[21];
            
            bool veto_hit = false;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_hit = true;
                    break;
                }
            }
            if (!veto_hit && (veto_energies[8] > TOP_VP_THRESHOLD || veto_energies[9] > TOP_VP_THRESHOLD)) {
                veto_hit = true;
            }
            
            if ((event_energy > MUON_ENERGY_THRESHOLD && veto_hit) ||
                (event_energy > MUON_ENERGY_THRESHOLD / 2 && veto_hit)) {
                is_muon = true;
            }
            
            event_times.push_back(current_time);
            is_muon_event.push_back(is_muon);
        }

        file->Close();
        delete file;
    }

    // Second pass: identify accidental events across all events
    size_t global_index = 0;
    for (const auto& inputFileName : inputFiles) {
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            continue;
        }

        TFile* file = TFile::Open(inputFileName.c_str());
        if (!file || file->IsZombie()) {
            continue;
        }

        TTree* tree = (TTree*)file->Get("tree");
        if (!tree) {
            file->Close();
            delete file;
            continue;
        }

        // Variables for tree branches
        Int_t eventID, nSamples[23], peakPosition[23], triggerBits;
        Short_t adcVal[23][45];
        Double_t baselineMean[23], baselineRMS[23], pulseH[23], area[23];
        Long64_t nsTime;

        // Set branch addresses
        tree->SetBranchAddress("eventID", &eventID);
        tree->SetBranchAddress("nSamples", nSamples);
        tree->SetBranchAddress("adcVal", adcVal);
        tree->SetBranchAddress("baselineMean", baselineMean);
        tree->SetBranchAddress("baselineRMS", baselineRMS);
        tree->SetBranchAddress("pulseH", pulseH);
        tree->SetBranchAddress("peakPosition", peakPosition);
        tree->SetBranchAddress("area", area);
        tree->SetBranchAddress("nsTime", &nsTime);
        tree->SetBranchAddress("triggerBits", &triggerBits);

        for (int iEnt = 0; iEnt < tree->GetEntries(); iEnt++) {
            tree->GetEntry(iEnt);
            global_index++;
            if (global_index == 0) continue; // Skip first event to ensure dt calculation
            
            // Skip beam-on events
            if (area[22] > EV61_THRESHOLD) continue;
            
            // Calculate event energy
            double event_energy = 0;
            for (int iChan = 0; iChan < N_PMTS; iChan++) {
                if (mu1[iChan] > 0) event_energy += area[PMT_CHANNEL_MAP[iChan]] / mu1[iChan];
            }
            
            // Get veto energies
            std::vector<double> veto_energies(10, 0);
            for (int iChan = 12; iChan < 20; iChan++) {
                veto_energies[iChan - 12] = area[iChan];
            }
            veto_energies[8] = area[20] * 1.07809;
            veto_energies[9] = area[21];
            
            // Check if event meets Michel-like criteria
            if (!isMichelLike(event_energy, veto_energies, triggerBits)) continue;
            
            // Check if this event is NOT preceded by a muon within MICHEL_DT_MAX
            bool preceded_by_muon = false;
            for (size_t j = global_index-1; j >= 0; j--) {
                double dt = event_times[global_index-1] - event_times[j];
                if (dt > MICHEL_DT_MAX) break; // Too far back in time
                if (is_muon_event[j]) {
                    preceded_by_muon = true;
                    break;
                }
            }
            
            if (!preceded_by_muon) {
                // This is an accidental event
                double dt = event_times[global_index-1] - event_times[global_index-2]; // Time since previous event
                if (dt > 0 && dt < MICHEL_DT_MAX) {
                    h_accidental_dt->Fill(dt);
                    h_accidental_energy->Fill(event_energy);
                    num_accidental++;
                }
            }
        }

        file->Close();
        delete file;
    }

    // Calculate accidental rate
    double exposure_time = (event_times.back() - first_event_time) / 1e6; // µs to s
    double accidental_rate = exposure_time > 0 ? num_accidental / exposure_time : 0;
    cout << "Accidental rate: " << accidental_rate << " Hz" << endl;

    // Fit accidental background with exponential
    TF1* f_accidental = new TF1("f_accidental", "[0]*exp(-x/[1])", 1.0, MICHEL_DT_MAX);
    f_accidental->SetParameters(100, 5.0);
    f_accidental->SetParNames("N0", "#tau_{accidental}");
    f_accidental->SetParLimits(0, 0, 1e6);
    f_accidental->SetParLimits(1, 0.1, 20.0);
    h_accidental_dt->Fit(f_accidental, "RQ");

    double tau_accidental = f_accidental->GetParameter(1);
    if (num_accidental < 50 || f_accidental->GetChisquare()/f_accidental->GetNDF() > 5) {
        tau_accidental = accidental_rate > 0 ? 1.0 / (accidental_rate * 1e-6) : 5.0;
        cout << "Accidental fit unreliable, using rate-based tau: " << tau_accidental << " µs" << endl;
    } else {
        cout << "Determined accidental tau: " << tau_accidental << " µs from " 
             << num_accidental << " background events" << endl;
    }

    // Plot the accidental background fit
    TCanvas* c_acc = new TCanvas("c_acc", "Accidental Background", 1200, 800);
    c_acc->SetMargin(0.12, 0.05, 0.12, 0.08);
    
    // Set histogram style
    setPlotStyle(h_accidental_dt);
    h_accidental_dt->SetLineColor(kBlack);
    h_accidental_dt->SetTitle("Accidental Background;dt (#mus);Counts / 0.08 #mus");
    
    // Set smart y-axis range
    double ymax_data = h_accidental_dt->GetBinContent(h_accidental_dt->GetMaximumBin());
    double ymax = ymax_data * 1.1;
    h_accidental_dt->SetMaximum(ymax);
    
    h_accidental_dt->Draw("HIST");
    
    f_accidental->SetLineColor(kRed);
    f_accidental->SetLineWidth(3);
    f_accidental->Draw("same");
    
    
    c_acc->Update();
    string plotName = OUTPUT_DIR + "/AccidentalBackgroundFit.png";
    c_acc->SaveAs(plotName.c_str());
    cout << "Saved accidental background plot: " << plotName << endl;

    // Plot accidental energy spectrum
    TCanvas* c_acc_energy = new TCanvas("c_acc_energy", "Accidental Energy Spectrum", 1200, 800);
    c_acc_energy->SetMargin(0.12, 0.05, 0.12, 0.08);
    
    // Set histogram style
    setPlotStyle(h_accidental_energy);
    h_accidental_energy->SetLineColor(kBlue);
    h_accidental_energy->SetTitle("Accidental Background Energy;Energy (p.e.);Counts / 8 p.e.");
    
    // Set smart y-axis range
    ymax_data = h_accidental_energy->GetBinContent(h_accidental_energy->GetMaximumBin());
    ymax = ymax_data * 1.1;
    h_accidental_energy->SetMaximum(ymax);
    
    h_accidental_energy->Draw("HIST");
    
    c_acc_energy->Update();
    plotName = OUTPUT_DIR + "/Accidental_Energy.png";
    c_acc_energy->SaveAs(plotName.c_str());
    cout << "Saved accidental energy plot: " << plotName << endl;

    // Cleanup
    delete c_acc;
    delete c_acc_energy;
    delete f_accidental;
    delete h_accidental_dt;
    delete h_accidental_energy;

    return tau_accidental;
}

// Utility functions
template<typename T>
double getAverage(const std::vector<T>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template<typename T>
double mostFrequent(const std::vector<T>& v) {
    if (v.empty()) return 0;
    std::map<T, int> count;
    for (const auto& val : v) count[val]++;
    T most_common = v[0];
    int max_count = 0;
    for (const auto& pair : count) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }
    return max_count > 1 ? most_common : getAverage(v);
}

template<typename T>
double variance(const std::vector<T>& v) {
    if (v.size() <= 1) return 0;
    double mean = getAverage(v);
    double sum = 0;
    for (const auto& val : v) {
        sum += (val - mean) * (val - mean);
    }
    return sum / (v.size() - 1);
}

// Create output directory
void createOutputDirectory(const string& dirName) {
    struct stat st;
    if (stat(dirName.c_str(), &st)) {
        if (mkdir(dirName.c_str(), 0755)) {
            cerr << "Error: Could not create directory " << dirName << endl;
            exit(1);
        }
        cout << "Created output directory: " << dirName << endl;
    } else {
        cout << "Output directory already exists: " << dirName << endl;
    }
}

// SPE calibration function
bool performCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cerr << "Error opening calibration file: " << calibFileName << endl;
        return false;
    }

    TTree *calibTree = (TTree*)calibFile->Get("tree");
    if (!calibTree) {
        cerr << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        delete calibFile;
        return false;
    }

    string speDir = OUTPUT_DIR + "/SPE_Fits";
    gSystem->mkdir(speDir.c_str(), kTRUE);

    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i + 1),
                             Form("PMT %d;ADC Counts;Events", i + 1), 150, -50, 400);
        histArea[i]->SetLineColor(kRed);
    }

    Int_t triggerBits;
    Double_t area[23];
    calibTree->SetBranchAddress("triggerBits", &triggerBits);
    calibTree->SetBranchAddress("area", area);

    Long64_t nEntries = calibTree->GetEntries();
    cout << "Processing " << nEntries << " calibration events from " << calibFileName << "..." << endl;

    for (Long64_t entry = 0; entry < nEntries; entry++) {
        calibTree->GetEntry(entry);
        if (triggerBits != 16) continue;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            histArea[pmt]->Fill(area[PMT_CHANNEL_MAP[pmt]]);
            nLEDFlashes[pmt]++;
        }
    }

    Int_t defaultErrorLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kError;

    // Create directory for individual PMT plots
    string individualPlotsDir = speDir + "/Individual";
    gSystem->mkdir(individualPlotsDir.c_str(), kTRUE);

    // Main canvas for combined view
    TCanvas *c_combined = new TCanvas("c_combined", "SPE Fits - Combined", 1200, 800);
    c_combined->Divide(4, 3);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);
    
    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i + 1 << " in " << calibFileName << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            delete histArea[i];
            continue;
        }

        c_combined->cd(i+1);
        
        TF1 *f1 = new TF1("f1", fitGauss, -50, 50, 3);
        f1->SetParameters(1500, 0, 25);
        f1->SetParNames("A0", "#mu_{0}", "#sigma_{0}");
        histArea[i]->Fit(f1, "Q", "", -50, 50);

        TF1 *f6 = new TF1("f6", six_fit_func, -50, 200, 6);
        f6->SetParameters(f1->GetParameter(0), f1->GetParameter(1), f1->GetParameter(2),
                          500, 100, 25);
        f6->SetParNames("A0", "#mu_{0}", "#sigma_{0}", "A1", "#mu_{1}", "#sigma_{1}");
        histArea[i]->Fit(f6, "Q", "", -50, 200);

        TF1 *f8 = new TF1("f8", eight_fit_func, -50, 400, 8);
        f8->SetParameters(f6->GetParameter(0), f6->GetParameter(1), f6->GetParameter(2), 
                          f6->GetParameter(3), f6->GetParameter(4), f6->GetParameter(5), 
                          200, 50);
        f8->SetParNames("A0", "#mu_{0}", "#sigma_{0}", "A1", "#mu_{1}", "#sigma_{1}", "A2", "A3");
        f8->SetLineColor(kBlue);
        histArea[i]->Fit(f8, "Q", "", -50, 400);

        mu1[i] = f8->GetParameter(4);
        mu1_err[i] = f8->GetParError(4);

        histArea[i]->Draw();
        f8->Draw("same");

        TLatex tex;
        tex.SetTextFont(42);
        tex.SetTextSize(0.04);
        tex.SetNDC();
        tex.DrawLatex(0.15, 0.85, Form("PMT %d", i+1));
        tex.DrawLatex(0.15, 0.80, Form("mu1 = %.2f #pm %.2f", mu1[i], mu1_err[i]));
        
        TCanvas *c_indiv = new TCanvas(Form("c_pmt%d", i+1), Form("PMT %d SPE Fit", i+1), 1200, 800);
        histArea[i]->Draw();
        f8->Draw("same");
        tex.DrawLatex(0.15, 0.85, Form("PMT %d", i+1));
        tex.DrawLatex(0.15, 0.80, Form("mu1 = %.2f #pm %.2f", mu1[i], mu1_err[i]));
        
        string indivPlotName = individualPlotsDir + Form("/PMT%d_SPE_Fit.png", i+1);
        c_indiv->SaveAs(indivPlotName.c_str());
        cout << "Saved individual plot: " << indivPlotName << endl;
        
        delete c_indiv;
        delete f1;
        delete f6;
        delete f8;
    }

    string combinedPlotName = speDir + "/SPE_Fits_Combined.png";
    c_combined->SaveAs(combinedPlotName.c_str());
    cout << "Saved combined SPE plot: " << combinedPlotName << endl;

    gErrorIgnoreLevel = defaultErrorLevel;
    
    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]) delete histArea[i];
    }
    delete c_combined;
    calibFile->Close();
    delete calibFile;
    return true;
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <calibration_file> <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }

    string calibFileName = argv[1];
    vector<string> inputFiles;
    for (int i = 2; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }

    // Create output directory
    createOutputDirectory(OUTPUT_DIR);

    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    // Check if calibration file exists
    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cerr << "Error: Calibration file " << calibFileName << " not found" << endl;
        return -1;
    }

    // Check if at least one input file exists
    bool anyInputFileExists = false;
    for (const auto& file : inputFiles) {
        if (!gSystem->AccessPathName(file.c_str())) {
            anyInputFileExists = true;
            break;
        }
    }
    if (!anyInputFileExists) {
        cerr << "Error: No input files found" << endl;
        return -1;
    }

    // Perform SPE calibration
    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    if (!performCalibration(calibFileName, mu1, mu1_err)) {
        cerr << "SPE calibration failed!" << endl;
        return -1;
    }

    // Print calibration results
    cout << "SPE Calibration Results (from " << calibFileName << "):\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }

    // Determine accidental background time constant using all input files
    FIXED_TAU_ACCIDENTAL = determineAccidentalTau(inputFiles, mu1);

    // Statistics counters
    int num_muons = 0;
    int num_michels = 0;
    int num_events = 0;

    // Map to track triggerBits counts
    std::map<int, int> trigger_counts;

    // Define histograms
    TH1D* h_muon_energy = new TH1D("muon_energy", "Muon Energy Distribution (with Michel Electrons);Energy (p.e.);Counts/100 p.e.", 550, -500, 5000);
    TH1D* h_michel_energy = new TH1D("michel_energy", "Michel Electron Energy Distribution;Energy (p.e.);Counts/8 p.e.", 100, 0, 800);
    TH1D* h_dt_michel = new TH1D("DeltaT", "Muon-Michel Time Difference;Time to Previous event(Muon)(#mus);Counts/0.08 #mus", 200, 0, MICHEL_DT_MAX);
    TH2D* h_energy_vs_dt = new TH2D("energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, 16, 200, 0, 1000);
    TH1D* h_side_vp_muon = new TH1D("side_vp_muon", "Side Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_top_vp_muon = new TH1D("top_vp_muon", "Top Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 1000);
    TH1D* h_trigger_bits = new TH1D("trigger_bits", "Trigger Bits Distribution;Trigger Bits;Counts", 36, 0, 36);

    // First pass: Fill histograms with fixed MICHEL_ENERGY_MIN
    for (const auto& inputFileName : inputFiles) {
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = new TFile(inputFileName.c_str());
        cout << "Processing file: " << inputFileName << endl;

        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Could not find tree in file: " << inputFileName << endl;
            f->Close();
            delete f;
            continue;
        }

        // Declaration of leaf types
        Int_t eventID, nSamples[23], peakPosition[23], triggerBits;
        Short_t adcVal[23][45];
        Double_t baselineMean[23], baselineRMS[23], pulseH[23], area[23];
        Long64_t nsTime;

        // Set branch addresses
        t->SetBranchAddress("eventID", &eventID);
        t->SetBranchAddress("nSamples", nSamples);
        t->SetBranchAddress("adcVal", adcVal);
        t->SetBranchAddress("baselineMean", baselineMean);
        t->SetBranchAddress("baselineRMS", baselineRMS);
        t->SetBranchAddress("pulseH", pulseH);
        t->SetBranchAddress("peakPosition", peakPosition);
        t->SetBranchAddress("area", area);
        t->SetBranchAddress("nsTime", &nsTime);
        t->SetBranchAddress("triggerBits", &triggerBits);

        int numEntries = t->GetEntries();
        cout << "Processing " << numEntries << " entries in " << inputFileName << endl;
        double last_muon_time = 0.0;
        std::set<double> michel_muon_times;
        std::vector<std::pair<double, double>> muon_candidates;
        std::vector<double> event_times;
        std::vector<bool> is_muon_event;

        // First pass: collect event times and muon flags
        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            double current_time = nsTime / 1000.0;
            event_times.push_back(current_time);
            
            // Calculate event energy
            double event_energy = 0;
            for (int iChan = 0; iChan < N_PMTS; iChan++) {
                if (mu1[iChan] > 0) event_energy += area[PMT_CHANNEL_MAP[iChan]] / mu1[iChan];
            }
            
            // Get veto energies
            std::vector<double> veto_energies(10, 0);
            for (int iChan = 12; iChan < 20; iChan++) {
                veto_energies[iChan - 12] = area[iChan];
            }
            veto_energies[8] = area[20] * 1.07809;
            veto_energies[9] = area[21];
            
            // Check if this is a muon event
            bool veto_hit = false;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_hit = true;
                    break;
                }
            }
            if (!veto_hit && (veto_energies[8] > TOP_VP_THRESHOLD || veto_energies[9] > TOP_VP_THRESHOLD)) {
                veto_hit = true;
            }
            
            bool is_muon = ((event_energy > MUON_ENERGY_THRESHOLD && veto_hit) ||
                           (event_energy > MUON_ENERGY_THRESHOLD / 2 && veto_hit));
            is_muon_event.push_back(is_muon);
        }

        // Second pass: identify Michel and muon events
        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;

            h_trigger_bits->Fill(triggerBits);
            trigger_counts[triggerBits]++;
            if (triggerBits < 0 || triggerBits > 36) {
                cout << "Warning: triggerBits = " << triggerBits << " out of histogram range (0-36) in file " << inputFileName << ", event " << eventID << endl;
            }

            struct pulse p;
            p.start = nsTime / 1000.0;
            p.end = nsTime / 1000.0;
            p.peak = 0;
            p.energy = 0;
            p.number = 0;
            p.single = false;
            p.beam = false;
            p.trigger = triggerBits;
            p.side_vp_energy = 0;
            p.top_vp_energy = 0;
            p.all_vp_energy = 0;
            p.last_muon_time = last_muon_time;
            p.is_muon = false;
            p.is_michel = false;

            std::vector<double> all_chan_start, all_chan_end, all_chan_peak, all_chan_energy;
            std::vector<double> side_vp_energy, top_vp_energy;
            std::vector<double> chan_starts_no_outliers;
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);
            bool pulse_at_end = false;
            int pulse_at_end_count = 0;
            std::vector<double> veto_energies(10, 0);

            for (int iChan = 0; iChan < 23; iChan++) {
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                if (iChan == 22) {
                    double ev61_energy = 0;
                    for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                        ev61_energy += h_wf.GetBinContent(iBin);
                    }
                    if (ev61_energy > EV61_THRESHOLD) {
                        p.beam = true;
                    }
                }

                std::vector<pulse_temp> pulses_temp;
                bool onPulse = false;
                int thresholdBin = 0, peakBin = 0;
                double peak = 0, pulseEnergy = 0;
                double allPulseEnergy = 0;

                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    double iBinContent = h_wf.GetBinContent(iBin);
                    if (iBin > 15) allPulseEnergy += iBinContent;

                    if (!onPulse && iBinContent >= PULSE_THRESHOLD) {
                        onPulse = true;
                        thresholdBin = iBin;
                        peakBin = iBin;
                        peak = iBinContent;
                        pulseEnergy = iBinContent;
                    } else if (onPulse) {
                        pulseEnergy += iBinContent;
                        if (peak < iBinContent) {
                            peak = iBinContent;
                            peakBin = iBin;
                        }
                        if (iBinContent < BS_UNCERTAINTY || iBin == ADCSIZE) {
                            pulse_temp pt;
                            pt.start = thresholdBin * 16.0 / 1000.0;
                            pt.peak = iChan <= 11 && mu1[iChan] > 0 ? peak / mu1[iChan] : peak;
                            pt.end = iBin * 16.0 / 1000.0;
                            for (int j = peakBin - 1; j >= 1 && h_wf.GetBinContent(j) > BS_UNCERTAINTY; j--) {
                                if (h_wf.GetBinContent(j) > peak * 0.1) {
                                    pt.start = j * 16.0 / 1000.0;
                                }
                                pulseEnergy += h_wf.GetBinContent(j);
                            }
                            if (iChan <= 11) {
                                pt.energy = mu1[iChan] > 0 ? pulseEnergy / mu1[iChan] : 0;
                                all_chan_start.push_back(pt.start);
                                all_chan_end.push_back(pt.end);
                                all_chan_peak.push_back(pt.peak);
                                all_chan_energy.push_back(pt.energy);
                                if (pt.energy > 1) p.number += 1;
                            }
                            pulses_temp.push_back(pt);
                            peak = 0;
                            peakBin = 0;
                            pulseEnergy = 0;
                            thresholdBin = 0;
                            onPulse = false;
                        }
                    }
                }

                if (iChan >= 12 && iChan <= 19) {
                    side_vp_energy.push_back(allPulseEnergy);
                    veto_energies[iChan - 12] = allPulseEnergy;
                } else if (iChan >= 20 && iChan <= 21) {
                    double factor = (iChan == 20) ? 1.07809 : 1.0;
                    top_vp_energy.push_back(allPulseEnergy * factor);
                    veto_energies[iChan - 12] = allPulseEnergy * factor;
                }

                if (iChan <= 11 && h_wf.GetBinContent(ADCSIZE) > 100) {
                    pulse_at_end_count++;
                    if (pulse_at_end_count >= 10) pulse_at_end = true;
                }

                h_wf.Reset();
            }

            p.start += mostFrequent(all_chan_start);
            p.end += mostFrequent(all_chan_end);
            p.energy = std::accumulate(all_chan_energy.begin(), all_chan_energy.end(), 0.0);
            p.peak = std::accumulate(all_chan_peak.begin(), all_chan_peak.end(), 0.0);
            p.side_vp_energy = std::accumulate(side_vp_energy.begin(), side_vp_energy.begin(), 0.0);
            p.top_vp_energy = std::accumulate(top_vp_energy.begin(), top_vp_energy.end(), 0.0);
            p.all_vp_energy = p.side_vp_energy + p.top_vp_energy;

            for (const auto& start : all_chan_start) {
                if (fabs(start - mostFrequent(all_chan_start)) < 10 * 16.0 / 1000.0) {
                    chan_starts_no_outliers.push_back(start);
                }
            }
            p.single = (variance(chan_starts_no_outliers) < 5 * 16.0 / 1000.0);

            bool veto_hit = false;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_hit = true;
                    break;
                }
            }
            if (!veto_hit && p.top_vp_energy > TOP_VP_THRESHOLD) veto_hit = true;

            if ((p.energy > MUON_ENERGY_THRESHOLD && veto_hit) ||
                (pulse_at_end && p.energy > MUON_ENERGY_THRESHOLD / 2 && veto_hit)) {
                p.is_muon = true;
                last_muon_time = p.start;
                num_muons++;
                muon_candidates.emplace_back(p.start, p.energy);
                h_side_vp_muon->Fill(p.side_vp_energy);
                h_top_vp_muon->Fill(p.top_vp_energy);
            }

            double dt = p.start - last_muon_time;
            
            bool is_michel_candidate = isMichelLike(p.energy, veto_energies, triggerBits) &&
                                      dt >= MICHEL_DT_MIN &&
                                      dt <= MICHEL_DT_MAX &&
                                      p.number >= 8;
            
            h_energy_vs_dt->Fill(dt, p.energy);

            bool is_michel_for_dt = is_michel_candidate && p.energy <= MICHEL_ENERGY_MAX_DT;

            if (is_michel_candidate) {
                p.is_michel = true;
                num_michels++;
                michel_muon_times.insert(last_muon_time);
                h_michel_energy->Fill(p.energy);
            }

            if (is_michel_for_dt) {
                h_dt_michel->Fill(dt);
            }

            p.last_muon_time = last_muon_time;
        }

        for (const auto& muon : muon_candidates) {
            if (michel_muon_times.find(muon.first) != michel_muon_times.end()) {
                h_muon_energy->Fill(muon.second);
            }
        }

        cout << "File " << inputFileName << " Statistics:\n";
        cout << "Total Events: " << num_events << "\n";
        cout << "Muons Detected: " << num_muons << "\n";
        cout << "Michel Electrons Detected: " << num_michels << "\n";
        cout << "------------------------\n";

        f->Close();
        delete f;
        num_events = 0;
        num_muons = 0;
        num_michels = 0;
    }

    // Fit Michel peak
    double michel_peak = fitMichelPeak(h_michel_energy);
    if (michel_peak > 0) {
        cout << "Michel peak found at " << michel_peak << " p.e., using MICHEL_ENERGY_MIN = " << MICHEL_ENERGY_MIN << " p.e." << endl;
    } else {
        cout << "Michel peak fit failed, using MICHEL_ENERGY_MIN = " << MICHEL_ENERGY_MIN << " p.e." << endl;
    }

    // Generate the Michel energy plot without threshold line
    plotMichelEnergy(h_michel_energy);

    // Fit the Michel dt distribution
    fitMichelDt(h_dt_michel);

    // Print triggerBits distribution
    cout << "Trigger Bits Distribution (all files):\n";
    for (const auto& pair : trigger_counts) {
        cout << "Trigger " << pair.first << ": " << pair.second << " events\n";
    }
    cout << "------------------------\n";

    // Generate remaining analysis plots
    TCanvas *c = new TCanvas("c", "Analysis Plots", 1200, 800);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);

    // Muon Energy
    c->Clear();
    c->SetMargin(0.12, 0.05, 0.12, 0.08);
    setPlotStyle(h_muon_energy);
    h_muon_energy->SetLineColor(kBlue);
    setSmartYRange(h_muon_energy);
    h_muon_energy->Draw("HIST");
    c->Update();
    string plotName = OUTPUT_DIR + "/Muon_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Energy vs dt
    c->Clear();
    c->SetMargin(0.12, 0.05, 0.12, 0.08);
    setPlotStyle(h_energy_vs_dt);
    h_energy_vs_dt->SetStats(0);
    h_energy_vs_dt->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Side Veto Muon
    c->Clear();
    c->SetMargin(0.12, 0.05, 0.12, 0.08);
    setPlotStyle(h_side_vp_muon);
    h_side_vp_muon->SetLineColor(kMagenta);
    setSmartYRange(h_side_vp_muon);
    h_side_vp_muon->Draw("HIST");
    c->Update();
    plotName = OUTPUT_DIR + "/Side_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Top Veto Muon
    c->Clear();
    c->SetMargin(0.12, 0.05, 0.12, 0.08);
    setPlotStyle(h_top_vp_muon);
    h_top_vp_muon->SetLineColor(kCyan);
    setSmartYRange(h_top_vp_muon);
    h_top_vp_muon->Draw("HIST");
    c->Update();
    plotName = OUTPUT_DIR + "/Top_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Trigger Bits Distribution
    c->Clear();
    c->SetMargin(0.12, 0.05, 0.12, 0.08);
    setPlotStyle(h_trigger_bits);
    h_trigger_bits->SetLineColor(kGreen);
    setSmartYRange(h_trigger_bits);
    h_trigger_bits->Draw("HIST");
    c->Update();
    plotName = OUTPUT_DIR + "/TriggerBits_Distribution.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Clean up
    delete h_muon_energy;
    delete h_michel_energy;
    delete h_dt_michel;
    delete h_energy_vs_dt;
    delete h_side_vp_muon;
    delete h_top_vp_muon;
    delete h_trigger_bits;
    delete c;

    cout << "Analysis complete. Results saved in " << OUTPUT_DIR << "/ (*.png)" << endl;
    return 0;
}
