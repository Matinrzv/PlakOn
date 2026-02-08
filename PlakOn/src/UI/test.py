from hezar.models import Model

model = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")
plate_text = model.predict("images/214_png.rf.aa4dc2956ae1f9cd56010672b89345e1.jpg")
print(plate_text)  # Persian text of mixed numbers and characters might not show correctly in the console
