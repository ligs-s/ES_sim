{


    //TString filename = "data/teflon_spec_0.05_diff_0.65/stype_point_iso_spos_\-116_sphere_1.root";
    TString filename = "data/teflon_spec_0.05_diff_0.65/stype_area_iso_spos_\-116_sphere_1.root";

    //TString filename = "data/stype_point_iso_spos_\-116_sphere_0.root";

    //TString filename = "data/stype_area_norm_spos_\-100_sphere_1.root";
    TFile* f = new TFile(filename, "read");
    tree = (TTree*)f->Get("tree");

    TCut cut = "abs(x_det)<3&&abs(z_det)<3";

    tree->SetAlias("r_gen", "sqrt(x_gen*x_gen+z_gen*z_gen)");
    tree->SetAlias("track_length", "t_det*300"); // t_det * 0.3 m/ns
    // source generation
    TCanvas* c1 = new TCanvas();
    c1->Divide(2,3);
    c1->cd(1);
    tree->Draw("r_gen*r_gen", cut);
    c1->cd(2);
    tree->Draw("y_gen", cut);
    c1->cd(3);
    tree->Draw("track_length:r_gen", cut, "colz");
    c1->cd(4);
    tree->Draw("y_gen:r_gen", cut);
    c1->cd(5);
    tree->Draw("theta_gen", cut);
    c1->cd(6);
    tree->Draw("z_gen:x_gen", cut);

    // detection distribution
    TCanvas* c2 = new TCanvas();
    c2->Divide(2,3);
    c2->cd(1);
    tree->Draw("theta_det");
    c2->cd(2);
    tree->Draw("theta_det>>hh", "abs(x_det)<3&&abs(z_det)<3");
    int bin1 = hh->FindBin(10);
    int nbins = hh->GetNbinsX();
    cout << "sphere enhancement factor: " << hh->Integral()/hh->Integral(1, bin1) << endl;
    c2->cd(3);
    tree->Draw("y_det");
    c2->cd(4);
    tree->Draw("z_det:x_det");
    c2->cd(5);
    tree->Draw("track_length:theta_det", "abs(x_det)<3&&abs(z_det)<3", "colz");
    c2->cd(6);
    tree->Draw("theta_gen:theta_det", "abs(x_det)<3&&abs(z_det)<3", "colz");


}
