vim.api.nvim_create_autocmd("FileType", {
	pattern = "python",
	group = vim.api.nvim_create_augroup("RvmPythonLsp", { clear = true }),
	callback = function()
		vim.lsp.start({
			name = "pylsp",
			cmd = { "pylsp" },
			settings = {
				pylsp = {
					plugins = {
						ruff = { enabled = true, formatEnabled = true },
						pycodestyle = { enabled = false },
						pyflakes = { enabled = false },
						mccabe = { enabled = false },
						autopep8 = { enabled = false },
						yapf = { enabled = false },
					},
				},
			},
		})
	end,
})

vim.api.nvim_create_autocmd("BufWritePre", {
	pattern = "*.py",
	callback = function()
		vim.lsp.buf.format({ async = false })
	end,
})
