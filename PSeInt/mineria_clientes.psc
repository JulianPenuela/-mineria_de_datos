// Guia 10 - Mineria de datos con IA (version PSeInt, 15 clientes de ejemplo)
// Fases: carga -> limpieza -> indicadores -> K-Means (k=3) -> perfil de grupos
// Variables por cliente: edad, compras, valor (millones $), frecuencia (compras/mes), inactivo (1=si, 0=no)
Algoritmo mineria_clientes
	Definir n, i, it, g, validos, mejor Como Entero
	Definir sumaValor, sumaCompras, inactivos, d, dmin, minF, maxF, minV, maxV Como Real
	Dimension edad[15], compras[15], valor[15], frec[15], inact[15], ok[15], grupo[15]
	Dimension px[15], py[15], cx[3], cy[3], sx[3], sy[3], cnt[3], sinact[3], sval[3], sfrec[3]
	n <- 15
	// ---- Fase 1: datos (hay errores a proposito: edad 999, valor -1 = faltante) ----
	edad[1]<-30;  compras[1]<-28; valor[1]<-1.7; frec[1]<-2.4; inact[1]<-0
	edad[2]<-27;  compras[2]<-31; valor[2]<-1.9; frec[2]<-2.6; inact[2]<-0
	edad[3]<-33;  compras[3]<-25; valor[3]<-1.5; frec[3]<-2.1; inact[3]<-0
	edad[4]<-29;  compras[4]<-30; valor[4]<-1.8; frec[4]<-2.5; inact[4]<-0
	edad[5]<-999; compras[5]<-27; valor[5]<-1.6; frec[5]<-2.3; inact[5]<-0
	edad[6]<-40;  compras[6]<-12; valor[6]<-1.8; frec[6]<-1.0; inact[6]<-0
	edad[7]<-38;  compras[7]<-13; valor[7]<-2.0; frec[7]<-1.1; inact[7]<-0
	edad[8]<-42;  compras[8]<-11; valor[8]<-1.6; frec[8]<-0.9; inact[8]<-0
	edad[9]<-45;  compras[9]<-12; valor[9]<- -1; frec[9]<-1.0; inact[9]<-0
	edad[10]<-39; compras[10]<-14; valor[10]<-2.1; frec[10]<-1.2; inact[10]<-1
	edad[11]<-50; compras[11]<-3; valor[11]<-0.8; frec[11]<-0.3; inact[11]<-1
	edad[12]<-47; compras[12]<-2; valor[12]<-0.6; frec[12]<-0.2; inact[12]<-1
	edad[13]<-52; compras[13]<-4; valor[13]<-1.0; frec[13]<-0.3; inact[13]<-1
	edad[14]<-49; compras[14]<-3; valor[14]<-0.7; frec[14]<-0.3; inact[14]<-0
	edad[15]<-55; compras[15]<-2; valor[15]<-0.5; frec[15]<-0.2; inact[15]<-1
	// ---- Fase 2: limpieza (se descartan registros con edad fuera de 18-100 o valor faltante) ----
	validos <- 0
	Para i <- 1 Hasta n Hacer
		Si edad[i] >= 18 Y edad[i] <= 100 Y valor[i] > 0 Entonces
			ok[i] <- 1
			validos <- validos + 1
		SiNo
			ok[i] <- 0
			Escribir "Registro ", i, " descartado (dato invalido o faltante)"
		FinSi
	FinPara
	Escribir "Registros validos: ", validos, " de ", n
	// ---- Fase 3: indicadores ----
	sumaValor <- 0; sumaCompras <- 0; inactivos <- 0
	Para i <- 1 Hasta n Hacer
		Si ok[i] = 1 Entonces
			sumaValor <- sumaValor + valor[i]
			sumaCompras <- sumaCompras + compras[i]
			inactivos <- inactivos + inact[i]
		FinSi
	FinPara
	Escribir "Ticket promedio (millones por compra): ", sumaValor / sumaCompras
	Escribir "% clientes inactivos: ", 100 * inactivos / validos
	// ---- Fase 4: K-Means con k=3, variables frecuencia y valor normalizadas (min-max) ----
	minF <- 1000; maxF <- -1000; minV <- 1000; maxV <- -1000
	Para i <- 1 Hasta n Hacer
		Si ok[i] = 1 Entonces
			Si frec[i] < minF Entonces minF <- frec[i]
			FinSi
			Si frec[i] > maxF Entonces maxF <- frec[i]
			FinSi
			Si valor[i] < minV Entonces minV <- valor[i]
			FinSi
			Si valor[i] > maxV Entonces maxV <- valor[i]
			FinSi
		FinSi
	FinPara
	Para i <- 1 Hasta n Hacer
		px[i] <- (frec[i] - minF) / (maxF - minF)
		py[i] <- (valor[i] - minV) / (maxV - minV)
	FinPara
	// centroides iniciales: frecuencia minima, media y maxima (deterministas)
	cx[1] <- 0; cy[1] <- 0.5
	cx[2] <- 0.5; cy[2] <- 0.5
	cx[3] <- 1; cy[3] <- 0.5
	Para it <- 1 Hasta 10 Hacer
		Para g <- 1 Hasta 3 Hacer
			sx[g] <- 0; sy[g] <- 0; cnt[g] <- 0
		FinPara
		Para i <- 1 Hasta n Hacer
			Si ok[i] = 1 Entonces
				mejor <- 1
				dmin <- (px[i]-cx[1])^2 + (py[i]-cy[1])^2
				Para g <- 2 Hasta 3 Hacer
					d <- (px[i]-cx[g])^2 + (py[i]-cy[g])^2
					Si d < dmin Entonces
						dmin <- d
						mejor <- g
					FinSi
				FinPara
				grupo[i] <- mejor
				sx[mejor] <- sx[mejor] + px[i]
				sy[mejor] <- sy[mejor] + py[i]
				cnt[mejor] <- cnt[mejor] + 1
			FinSi
		FinPara
		Para g <- 1 Hasta 3 Hacer
			Si cnt[g] > 0 Entonces // grupo vacio: se conserva el centroide
				cx[g] <- sx[g] / cnt[g]
				cy[g] <- sy[g] / cnt[g]
			FinSi
		FinPara
	FinPara
	// ---- Fases 5-6: perfil de cada grupo (en unidades originales) ----
	Para g <- 1 Hasta 3 Hacer
		sfrec[g] <- 0; sval[g] <- 0; sinact[g] <- 0; cnt[g] <- 0
	FinPara
	Para i <- 1 Hasta n Hacer
		Si ok[i] = 1 Entonces
			g <- grupo[i]
			cnt[g] <- cnt[g] + 1
			sfrec[g] <- sfrec[g] + frec[i]
			sval[g] <- sval[g] + valor[i]
			sinact[g] <- sinact[g] + inact[i]
		FinSi
	FinPara
	Escribir "----- PERFIL DE GRUPOS -----"
	Para g <- 1 Hasta 3 Hacer
		Si cnt[g] > 0 Entonces
			Escribir "Grupo ", g, ": ", cnt[g], " clientes | frecuencia media ", sfrec[g]/cnt[g], " | valor medio ", sval[g]/cnt[g], " | % inactivos ", 100*sinact[g]/cnt[g]
		SiNo
			Escribir "Grupo ", g, ": vacio"
		FinSi
	FinPara
	// ---- Fase 7: recomendacion automatica segun frecuencia del grupo ----
	Para g <- 1 Hasta 3 Hacer
		Si cnt[g] > 0 Entonces
			Si sfrec[g]/cnt[g] >= 2 Entonces
				Escribir "Grupo ", g, " (frecuentes): venta cruzada para subir el ticket."
			SiNo
				Si sfrec[g]/cnt[g] >= 0.7 Entonces
					Escribir "Grupo ", g, " (intermedios): programa de fidelizacion."
				SiNo
					Escribir "Grupo ", g, " (ocasionales): campana de reactivacion."
				FinSi
			FinSi
		FinSi
	FinPara
FinAlgoritmo
