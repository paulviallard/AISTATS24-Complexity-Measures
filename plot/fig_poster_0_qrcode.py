import qrcode

link = "https://arxiv.org/pdf/2402.13285.pdf"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(link)
qr.make(fit=True)
qr_image = qr.make_image(fill_color="black", back_color="white")
qr_image.save("figures/fig_poster_0_qrcode.png")
