void plot(){


    TString filename = "test_1.root";
    TFile* f = new TFile(filename, "read");
    tree = (TTree*)f->Get("tree");

    tree->SetAlias("r_gen", "sqrt(x_gen*x_gen+z_gen*z_gen)");
    tree->SetAlias("track_length", "t_det*300"); // t_det * 0.3 m/ns
    //tree->Draw("r_gen");
    //tree->Draw("y_gen");
    tree->Draw("track_length:r_gen", "", "colz");


}
